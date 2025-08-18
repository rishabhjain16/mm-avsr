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
    from transformers import LlamaModel, LlamaForCausalLM, LlamaConfig
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
        
        # Store model name
        self.model_name = model_name
        
        # Load LLaMA configuration first
        try:
            from espnet.nets.pytorch_backend.model_cache_utils import (
                load_model_from_path_or_download, get_model_path, check_model_exists_locally, log_model_info
            )
            
            # Load config from the same location
            model_path = get_model_path(model_name)
            if check_model_exists_locally(model_name):
                config = LlamaConfig.from_pretrained(model_path, local_files_only=True)
            else:
                config = LlamaConfig.from_pretrained(model_name)
            
            self.hidden_size = config.hidden_size
            
            # Load model with or without quantization based on use_lora flag
            if use_lora:
                self._load_model_with_qlora(model_name, lora_r, lora_alpha, lora_dropout)
            else:
                # Load model normally
                self.llama = load_model_from_path_or_download(LlamaModel, model_name)
            
            log_model_info(model_name, "LLaMA")
        except Exception as e:
            logging.error(f"Failed to load LLaMA model {model_name}: {e}")
            raise
        
        # Freeze base model if requested
        if freeze_base_model:
            for param in self.llama.parameters():
                param.requires_grad = False
        
        # Projection layer to align encoder features with LLaMA hidden size
        self.encoder_projection = nn.Linear(encoder_dim, self.hidden_size)
        
        # Output projection layer
        self.output_layer = nn.Linear(self.hidden_size, odim)
        
        # Log the configuration
        logging.info(f"LLaMA decoder initialized for simple language modeling")
    
    def _load_model_with_qlora(self, model_name: str, r: int, alpha: int, dropout: float):
        """Load model with QLoRA from the start - no double loading."""
        if not PEFT_AVAILABLE:
            logging.warning("peft not available. Loading model without LoRA.")
            from espnet.nets.pytorch_backend.model_cache_utils import load_model_from_path_or_download
            self.llama = load_model_from_path_or_download(LlamaModel, model_name)
            return
        
        from transformers import BitsAndBytesConfig
        from espnet.nets.pytorch_backend.model_cache_utils import (
            get_model_path, check_model_exists_locally
        )
        
        try:
            # Create quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                torch_dtype=torch.float16
            )
            
            # Load model with quantization directly
            model_path = get_model_path(model_name)
            if check_model_exists_locally(model_name):
                self.llama = LlamaModel.from_pretrained(
                    model_path, 
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16,
                    local_files_only=True,
                    device_map="auto"
                )
            else:
                self.llama = LlamaModel.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            
            # Apply LoRA
            lora_config = LoraConfig(
                r=r,
                lora_alpha=alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            
            self.llama = get_peft_model(self.llama, lora_config)
            logging.info(f"Applied QLoRA with r={r}, alpha={alpha}, dropout={dropout}")
            
        except Exception as e:
            logging.error(f"Failed to load model with QLoRA: {e}")
            logging.info("Falling back to regular model loading")
            
            # Fallback to regular loading
            from espnet.nets.pytorch_backend.model_cache_utils import load_model_from_path_or_download
            self.llama = load_model_from_path_or_download(LlamaModel, model_name)
            
            # Try regular LoRA
            try:
                lora_config = LoraConfig(
                    r=r,
                    lora_alpha=alpha,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                    lora_dropout=dropout,
                    bias="none",
                    task_type="FEATURE_EXTRACTION",
                )
                
                self.llama = get_peft_model(self.llama, lora_config)
                logging.info(f"Applied regular LoRA with r={r}, alpha={alpha}, dropout={dropout}")
            except Exception as lora_e:
                logging.error(f"Failed to apply LoRA: {lora_e}")
                logging.info("Using model without LoRA")
    
    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """Forward decoder.
        
        Args:
            tgt (torch.Tensor): input token ids, int64 (batch, maxlen_out)
            tgt_mask (torch.Tensor): input token mask, (batch, maxlen_out)
            memory (torch.Tensor): encoded memory, float32 (batch, maxlen_in, feat) - already projected to LLaMA dims
            memory_mask (torch.Tensor): encoded memory mask, (batch, maxlen_in)
        
        Returns:
            torch.Tensor: decoded token score before softmax (batch, maxlen_out, odim)
            torch.Tensor: target mask (batch, maxlen_out)
        """
        batch_size, seq_len = tgt.shape
        
        # Get LLaMA embeddings for target tokens
        inputs_embeds = self.llama.embed_tokens(tgt)  # (batch, maxlen_out, hidden_size)
        
        # Simple approach: Prepend encoder features to the input sequence
        # This allows LLaMA to attend to encoder information naturally
        if memory is not None:
            # Project encoder features to LLaMA dimensions
            memory_proj = self.encoder_projection(memory)  # (batch, maxlen_in, hidden_size)
            # Concatenate encoder features with target embeddings
            inputs_embeds = torch.cat([memory_proj, inputs_embeds], dim=1)  # (batch, maxlen_in + maxlen_out, hidden_size)
            
            # Update sequence length
            total_seq_len = inputs_embeds.size(1)
        else:
            total_seq_len = seq_len
        
        # Create causal attention mask
        causal_mask = subsequent_mask(total_seq_len, device=tgt.device)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        
        # Create position IDs
        position_ids = torch.arange(total_seq_len, device=tgt.device).unsqueeze(0).expand(batch_size, -1)
        
        # Forward through LLaMA
        llama_outputs = self.llama(
            input_ids=None,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        
        hidden_states = llama_outputs.last_hidden_state
        
        # Extract only the target token representations (skip encoder part)
        if memory is not None:
            memory_len = memory.size(1)
            hidden_states = hidden_states[:, memory_len:, :]  # (batch, maxlen_out, hidden_size)
        
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