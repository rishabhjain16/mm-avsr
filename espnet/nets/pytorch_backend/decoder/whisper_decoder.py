#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 ESPnet Contributors
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Whisper decoder definition."""

from typing import Any, List, Tuple, Optional
import logging

import torch
import torch.nn as nn
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask

try:
    from transformers import WhisperModel, WhisperConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. WhisperDecoder will not work.")


class WhisperDecoder(BatchScorerInterface, torch.nn.Module):
    """Whisper decoder module adapted for cross-attention with visual/audio features.

    This decoder uses the Whisper decoder architecture but adapts it to work with
    external encoder features (visual/audio) while maintaining compatibility with
    the existing CTC + attention loss training framework.

    Args:
        odim (int): output dimension (vocabulary size)
        model_name (str): HuggingFace Whisper model name
        encoder_dim (int): encoder output dimension for cross-attention
        freeze_base_model (bool): whether to freeze the base Whisper model
        use_multilingual (bool): whether to use multilingual Whisper capabilities
    """

    def __init__(
        self,
        odim: int,
        model_name: str = "openai/whisper-base",
        encoder_dim: int = 512,
        freeze_base_model: bool = False,
        use_multilingual: bool = False,
    ):
        """Construct a WhisperDecoder object."""
        super(WhisperDecoder, self).__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for WhisperDecoder. "
                "Please install it with: pip install transformers"
            )
        
        self.odim = odim
        self.model_name = model_name
        self.encoder_dim = encoder_dim
        self.use_multilingual = use_multilingual
        
        # Load Whisper configuration and model
        try:
            from espnet.nets.pytorch_backend.model_cache_utils import (
                load_model_from_path_or_download, get_model_path, check_model_exists_locally, log_model_info
            )
            
            # Load model from local path or download
            self.whisper = load_model_from_path_or_download(WhisperModel, model_name)
            
            # Load config from the same location
            model_path = get_model_path(model_name)
            if check_model_exists_locally(model_name):
                config = WhisperConfig.from_pretrained(model_path, local_files_only=True)
            else:
                config = WhisperConfig.from_pretrained(model_name)
            
            self.hidden_size = config.d_model
            self.num_layers = config.decoder_layers
            log_model_info(model_name, "Whisper decoder")
        except Exception as e:
            logging.error(f"Failed to load Whisper model {model_name}: {e}")
            raise
        
        # Extract only the decoder part
        self.whisper_decoder = self.whisper.decoder
        
        # Freeze base model if requested
        if freeze_base_model:
            for param in self.whisper_decoder.parameters():
                param.requires_grad = False
        
        # Projection layer to align external encoder features with Whisper hidden size
        self.encoder_projection = nn.Linear(encoder_dim, self.hidden_size)
        
        # Output projection layer (adapt to target vocabulary if different)
        if odim != config.vocab_size:
            self.output_layer = nn.Linear(self.hidden_size, odim)
        else:
            self.output_layer = None  # Use Whisper's original output projection
        
        # Layer norm for encoder features
        self.encoder_layer_norm = nn.LayerNorm(self.hidden_size)
        
        logging.info(f"Initialized WhisperDecoder with model {model_name}")
        logging.info(f"Hidden size: {self.hidden_size}, Encoder dim: {encoder_dim}")
        logging.info(f"Output dim: {odim}, Vocab size: {config.vocab_size}")
    
    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """Forward decoder.
        
        Args:
            tgt (torch.Tensor): input token ids, int64 (batch, maxlen_out)
            tgt_mask (torch.Tensor): input token mask, (batch, maxlen_out)
            memory (torch.Tensor): encoded memory, float32 (batch, maxlen_in, feat)
            memory_mask (torch.Tensor): encoded memory mask, (batch, maxlen_in)
        
        Returns:
            torch.Tensor: decoded token score before softmax (batch, maxlen_out, odim)
            torch.Tensor: target mask (batch, maxlen_out)
        """
        batch_size, seq_len = tgt.shape
        
        # Project external encoder features to Whisper hidden size
        encoder_hidden_states = self.encoder_projection(memory)  # (batch, maxlen_in, hidden_size)
        encoder_hidden_states = self.encoder_layer_norm(encoder_hidden_states)
        
        # Convert masks to attention mask format expected by Whisper
        # Whisper expects attention_mask where 1 = attend, 0 = don't attend
        encoder_attention_mask = None
        if memory_mask is not None:
            # memory_mask is typically 1 for valid positions, 0 for padding
            # Convert to the format expected by Whisper (invert if needed)
            encoder_attention_mask = memory_mask.float()
        
        # Create decoder attention mask (causal mask)
        decoder_attention_mask = None
        if tgt_mask is not None:
            decoder_attention_mask = tgt_mask.float()
        
        # Forward through Whisper decoder
        try:
            decoder_outputs = self.whisper_decoder(
                input_ids=tgt,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            
            hidden_states = decoder_outputs.last_hidden_state  # (batch, maxlen_out, hidden_size)
            
        except Exception as e:
            logging.error(f"Error in Whisper decoder forward pass: {e}")
            # Fallback: use a simple approach
            hidden_states = self._simple_forward(tgt, encoder_hidden_states, encoder_attention_mask)
        
        # Apply output projection if needed
        if self.output_layer is not None:
            output = self.output_layer(hidden_states)
        else:
            # Use Whisper's original output projection
            output = self.whisper.proj_out(hidden_states) if hasattr(self.whisper, 'proj_out') else hidden_states
        
        return output, tgt_mask
    
    def _simple_forward(self, tgt, encoder_hidden_states, encoder_attention_mask):
        """Simple fallback forward pass."""
        # Get embeddings
        inputs_embeds = self.whisper_decoder.embed_tokens(tgt)
        
        # Add positional embeddings
        if hasattr(self.whisper_decoder, 'embed_positions'):
            positions = self.whisper_decoder.embed_positions(tgt)
            inputs_embeds = inputs_embeds + positions
        
        # Apply dropout
        hidden_states = self.whisper_decoder.dropout(inputs_embeds)
        
        # Process through decoder layers
        for layer in self.whisper_decoder.layers:
            layer_outputs = layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            hidden_states = layer_outputs[0]
        
        # Apply final layer norm
        if hasattr(self.whisper_decoder, 'layer_norm'):
            hidden_states = self.whisper_decoder.layer_norm(hidden_states)
        
        return hidden_states
    
    def forward_one_step(self, tgt, tgt_mask, memory, memory_mask=None, cache=None):
        """Forward one step for beam search.
        
        Args:
            tgt (torch.Tensor): input token ids, int64 (batch, maxlen_out)
            tgt_mask (torch.Tensor): input token mask, (batch, maxlen_out)
            memory (torch.Tensor): encoded memory, float32 (batch, maxlen_in, feat)
            memory_mask (torch.Tensor): encoded memory mask, (batch, maxlen_in)
            cache (List[torch.Tensor]): cached states from previous steps
        
        Returns:
            torch.Tensor: log probabilities for next token (batch, odim)
            List[torch.Tensor]: updated cache
        """
        # Project encoder features
        encoder_hidden_states = self.encoder_projection(memory)
        encoder_hidden_states = self.encoder_layer_norm(encoder_hidden_states)
        
        # Convert masks
        encoder_attention_mask = memory_mask.float() if memory_mask is not None else None
        decoder_attention_mask = tgt_mask.float() if tgt_mask is not None else None
        
        # For beam search, we typically want to use caching for efficiency
        # For now, we'll use a simplified approach
        try:
            decoder_outputs = self.whisper_decoder(
                input_ids=tgt,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=True,
                past_key_values=cache,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            
            hidden_states = decoder_outputs.last_hidden_state
            new_cache = decoder_outputs.past_key_values
            
        except Exception as e:
            logging.warning(f"Error in cached forward pass: {e}. Using full forward.")
            output, _ = self.forward(tgt, tgt_mask, memory, memory_mask)
            hidden_states = output
            new_cache = cache
        
        # Get the last token's output
        last_hidden = hidden_states[:, -1, :]  # (batch, hidden_size)
        
        # Apply output projection
        if self.output_layer is not None:
            last_output = self.output_layer(last_hidden)
        else:
            last_output = self.whisper.proj_out(last_hidden) if hasattr(self.whisper, 'proj_out') else last_hidden
        
        # Apply log softmax
        log_probs = torch.log_softmax(last_output, dim=-1)
        
        return log_probs, new_cache
    
    def score(self, ys, state, x):
        """Score function for beam search."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, x.unsqueeze(0), cache=state
        )
        return logp.squeeze(0), state
    
    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch for beam search.
        
        Args:
            ys (torch.Tensor): prefix tokens (n_batch, ylen)
            states (List[Any]): scorer states for prefix tokens
            xs (torch.Tensor): encoder features (n_batch, xlen, n_feat)
        
        Returns:
            Tuple[torch.Tensor, List[Any]]: scores for next token (n_batch, n_vocab)
                and next state list
        """
        # Create causal mask
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        ys_mask = ys_mask.expand(ys.size(0), -1, -1)
        
        # For batch scoring, we need to handle states properly
        # This is a simplified implementation
        batch_size = ys.size(0)
        all_logp = []
        new_states = []
        
        for i in range(batch_size):
            single_ys = ys[i:i+1]  # (1, ylen)
            single_xs = xs[i:i+1]  # (1, xlen, feat)
            single_mask = ys_mask[i:i+1]  # (1, ylen, ylen)
            single_state = states[i] if states and len(states) > i else None
            
            logp, new_state = self.forward_one_step(
                single_ys, single_mask, single_xs, cache=single_state
            )
            
            all_logp.append(logp)
            new_states.append(new_state)
        
        # Concatenate results
        batch_logp = torch.cat(all_logp, dim=0)  # (batch_size, vocab_size)
        
        return batch_logp, new_states