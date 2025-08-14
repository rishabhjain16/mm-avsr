#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 ESPnet Contributors
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""LLaMA decoder definition."""

from typing import Any, List, Tuple, Optional
import logging

import torch
import torch.nn as nn
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask

try:
    from transformers import LlamaModel, LlamaConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. LLaMADecoder will not work.")

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("peft not available. LoRA fine-tuning will not work.")


class LLaMADecoder(BatchScorerInterface, torch.nn.Module):
    """LLaMA decoder module adapted for encoder-decoder attention.

    This decoder adapts the LLaMA model for use in encoder-decoder architectures
    by adding cross-attention layers and maintaining compatibility with the
    existing beam search interface.

    Args:
        odim (int): output dimension (vocabulary size)
        model_name (str): HuggingFace model name for LLaMA
        use_lora (bool): whether to use LoRA fine-tuning
        lora_r (int): LoRA rank
        lora_alpha (int): LoRA alpha parameter
        lora_dropout (float): LoRA dropout rate
        encoder_dim (int): encoder output dimension for cross-attention
        freeze_base_model (bool): whether to freeze the base LLaMA model
    """

    def __init__(
        self,
        odim: int,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        encoder_dim: int = 512,
        freeze_base_model: bool = True,
    ):
        """Construct a LLaMADecoder object."""
        super(LLaMADecoder, self).__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for LLaMADecoder. "
                "Please install it with: pip install transformers"
            )
        
        self.odim = odim
        self.model_name = model_name
        self.use_lora = use_lora
        self.encoder_dim = encoder_dim
        
        # Load LLaMA configuration and model
        try:
            config = LlamaConfig.from_pretrained(model_name)
            self.llama = LlamaModel.from_pretrained(model_name, config=config)
            self.hidden_size = config.hidden_size
        except Exception as e:
            logging.error(f"Failed to load LLaMA model {model_name}: {e}")
            raise
        
        # Freeze base model if requested
        if freeze_base_model:
            for param in self.llama.parameters():
                param.requires_grad = False
        
        # Add cross-attention layers for encoder-decoder attention
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=config.num_attention_heads,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(config.num_hidden_layers)
        ])
        
        # Projection layer to align encoder features with LLaMA hidden size
        self.encoder_projection = nn.Linear(encoder_dim, self.hidden_size)
        
        # Output projection layer
        self.output_layer = nn.Linear(self.hidden_size, odim)
        
        # Apply LoRA if requested
        if use_lora:
            if not PEFT_AVAILABLE:
                logging.warning(
                    "peft not available. Falling back to standard training."
                )
            else:
                self._apply_lora(lora_r, lora_alpha, lora_dropout)
    
    def _apply_lora(self, r: int, alpha: int, dropout: float):
        """Apply LoRA to the LLaMA model."""
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        self.llama = get_peft_model(self.llama, lora_config)
        logging.info(f"Applied LoRA with r={r}, alpha={alpha}, dropout={dropout}")
    
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
        
        # Project encoder features to LLaMA hidden size
        memory_proj = self.encoder_projection(memory)  # (batch, maxlen_in, hidden_size)
        
        # Get LLaMA embeddings
        inputs_embeds = self.llama.embed_tokens(tgt)  # (batch, maxlen_out, hidden_size)
        
        # Create attention mask for causal attention
        causal_mask = subsequent_mask(seq_len, device=tgt.device)
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Process through LLaMA layers with cross-attention
        hidden_states = inputs_embeds
        
        for i, (layer, cross_attn) in enumerate(zip(self.llama.layers, self.cross_attention_layers)):
            # Self-attention through LLaMA layer
            layer_outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]
            
            # Cross-attention with encoder memory
            cross_attn_output, _ = cross_attn(
                query=hidden_states,
                key=memory_proj,
                value=memory_proj,
                key_padding_mask=memory_mask if memory_mask is not None else None,
                need_weights=False,
            )
            
            # Residual connection
            hidden_states = hidden_states + cross_attn_output
        
        # Apply final layer norm
        hidden_states = self.llama.norm(hidden_states)
        
        # Project to output vocabulary
        output = self.output_layer(hidden_states)
        
        return output, tgt_mask
    
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
        # For simplicity, we'll use the full forward pass
        # In a production implementation, you'd want to optimize this with proper caching
        output, _ = self.forward(tgt, tgt_mask, memory, memory_mask)
        
        # Get the last token's output
        last_output = output[:, -1, :]  # (batch, odim)
        
        # Apply log softmax
        log_probs = torch.log_softmax(last_output, dim=-1)
        
        # For now, return empty cache (could be optimized with KV caching)
        new_cache = cache if cache is not None else []
        
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
        
        # Forward pass
        logp, new_states = self.forward_one_step(ys, ys_mask, xs, cache=states)
        
        return logp, new_states