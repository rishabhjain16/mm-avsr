#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 ESPnet Contributors
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Large Language Model decoder definition."""

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
    logging.warning("transformers not available. LLMDecoder will not work.")

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("peft not available. LoRA fine-tuning will not work.")


class LLMDecoder(BatchScorerInterface, torch.nn.Module):
    """Large Language Model decoder adapted for encoder-decoder attention.

    This decoder adapts large language models for use in encoder-decoder architectures
    by adding cross-attention layers and maintaining compatibility with the
    existing beam search interface.

    Args:
        odim (int): output dimension (vocabulary size)
        model_name (str): HuggingFace model name for the language model
        use_lora (bool): whether to use LoRA fine-tuning
        lora_r (int): LoRA rank
        lora_alpha (int): LoRA alpha parameter
        lora_dropout (float): LoRA dropout rate
        encoder_dim (int): encoder output dimension for cross-attention
        freeze_base_model (bool): whether to freeze the base language model
        use_instructions (bool): whether to use instruction prompts
        instruction_prompt (str): instruction prompt text
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
        use_instructions: bool = False,
    ):
        """Construct a LLMDecoder object."""
        super(LLMDecoder, self).__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for LLMDecoder. "
                "Please install it with: pip install transformers"
            )
        
        self.odim = odim
        self.model_name = model_name
        self.use_lora = use_lora
        self.encoder_dim = encoder_dim
        self.use_instructions = use_instructions
        
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
        
        # Simple instruction handling
        if self.use_instructions:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Will be set based on modality later
            self.instruction_prompt = None
            self.instruction_tokens = None
            
            logging.info("Instruction tuning enabled (prompt will be set based on modality)")
        
        # Move additional layers to the same device as the main model
        if hasattr(self.llama, 'device') and self.llama.device != torch.device('cpu'):
            device = self.llama.device
        else:
            # For QLoRA models, get device from the first parameter
            device = next(self.llama.parameters()).device
        
        self.encoder_projection = self.encoder_projection.to(device)
        self.output_layer = self.output_layer.to(device)
        
        # Log the configuration
        config_msg = f"LLaMA decoder initialized"
        if self.use_instructions:
            config_msg += f" with instruction tuning"
        logging.info(config_msg)
    
    def _load_model_with_qlora(self, model_name: str, r: int, alpha: int, dropout: float):
        """Load model with QLoRA - fail fast if dependencies missing."""
        if not PEFT_AVAILABLE:
            raise ImportError("peft package required for QLoRA. Install with: pip install peft")
        
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError("bitsandbytes required for QLoRA. Install with: pip install bitsandbytes")
        
        from espnet.nets.pytorch_backend.model_cache_utils import (
            get_model_path, check_model_exists_locally
        )
        
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
        
        # Apply LoRA - use FEATURE_EXTRACTION for LlamaModel
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
    
    def set_instruction_prompt(self, modality):
        """Set instruction prompt based on modality."""
        if not self.use_instructions:
            return
        
        # Simple modality-based prompts
        if modality == "audio":
            prompt = "Transcribe audio features to text."
        elif modality == "video":
            prompt = "Transcribe video features to text."
        elif modality == "multimodal":
            prompt = "Transcribe audio and video features to text."
        else:
            prompt = "Transcribe features to text."
        
        self.instruction_prompt = prompt
        self.instruction_tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.squeeze(0)
        
        logging.info(f"Instruction prompt set for {modality}: '{prompt}'")
    
    def get_instruction_embeddings(self, batch_size, device):
        """Get instruction embeddings for the batch."""
        if not self.use_instructions or not hasattr(self, 'instruction_tokens'):
            return None
        
        # Expand instruction tokens for batch
        instruction_ids = self.instruction_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        
        # Get embeddings
        return self.llama.embed_tokens(instruction_ids)
    
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
        
        # Build input sequence: [INSTRUCTION] + [ENCODER_FEATURES] + [TARGET_TOKENS]
        sequence_parts = []
        prefix_len = 0
        
        # Add instruction if enabled
        if self.use_instructions:
            instruction_embeds = self.get_instruction_embeddings(batch_size, tgt.device)
            if instruction_embeds is not None:
                sequence_parts.append(instruction_embeds)
                prefix_len += instruction_embeds.size(1)
        
        # Add encoder features
        if memory is not None:
            memory_proj = self.encoder_projection(memory)
            sequence_parts.append(memory_proj)
            prefix_len += memory_proj.size(1)
        
        # Add target embeddings
        sequence_parts.append(inputs_embeds)
        
        # Concatenate sequence
        inputs_embeds = torch.cat(sequence_parts, dim=1)
        total_seq_len = inputs_embeds.size(1)
        
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
        
        # Extract target token representations (skip prefix)
        if prefix_len > 0:
            hidden_states = hidden_states[:, prefix_len:, :]
        
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