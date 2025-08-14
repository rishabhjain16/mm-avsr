#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""QLoRA utilities for memory-efficient training of large models."""

import torch
import torch.nn as nn
from typing import List, Iterator, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def apply_qlora(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.1,
    use_4bit: bool = True,
    **kwargs
) -> nn.Module:
    """Apply QLoRA (4-bit quantization + LoRA) to a model.
    
    Args:
        model (nn.Module): The model to apply QLoRA to
        target_modules (List[str], optional): List of module names to target for LoRA.
            If None, will target common linear layers.
        r (int): LoRA rank parameter
        alpha (int): LoRA alpha parameter
        dropout (float): LoRA dropout rate
        use_4bit (bool): Whether to use 4-bit quantization
        **kwargs: Additional arguments
        
    Returns:
        nn.Module: Model with QLoRA applied
        
    Raises:
        ImportError: If required packages (peft, bitsandbytes) are not installed
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        if use_4bit:
            import bitsandbytes as bnb
    except ImportError as e:
        raise ImportError(
            f"QLoRA requires 'peft' and 'bitsandbytes' packages. "
            f"Install with: pip install peft bitsandbytes. Error: {e}"
        )
    
    # Find actual target modules in the model
    actual_target_modules = _find_target_modules(model, target_modules)
    
    if not actual_target_modules:
        logger.warning("No target modules found for LoRA. Skipping QLoRA application.")
        return model
    
    # Apply 4-bit quantization if requested
    if use_4bit:
        logger.info("Applying 4-bit quantization to model")
        model = _apply_4bit_quantization(model)
    
    # Configure LoRA with actual target modules
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=actual_target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,  # Default task type
    )
    
    # Apply LoRA
    logger.info(f"Applying LoRA with r={r}, alpha={alpha}, dropout={dropout}")
    logger.info(f"Target modules found: {actual_target_modules}")
    model = get_peft_model(model, lora_config)
    
    return model


def _find_target_modules(model: nn.Module, target_modules: Optional[List[str]] = None) -> List[str]:
    """Find actual target modules that exist in the model."""
    
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Get all module names in the model
    all_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Extract the last part of the module name
            module_name = name.split('.')[-1]
            all_module_names.add(module_name)
    
    # Find intersection of target modules and actual modules
    actual_targets = []
    for target in target_modules:
        if target in all_module_names:
            actual_targets.append(target)
    
    # If no standard targets found, try to find any linear layers
    if not actual_targets:
        logger.info("No standard target modules found, searching for any linear layers...")
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module_name = name.split('.')[-1]
                if module_name not in actual_targets:
                    actual_targets.append(module_name)
                # Limit to avoid too many targets
                if len(actual_targets) >= 10:
                    break
    
    return actual_targets


def _apply_4bit_quantization(model: nn.Module) -> nn.Module:
    """Apply 4-bit quantization to linear layers in the model.
    
    Args:
        model (nn.Module): Model to quantize
        
    Returns:
        nn.Module: Quantized model
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError("4-bit quantization requires 'bitsandbytes' package")
    
    # Replace linear layers with 4-bit quantized versions
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Create 4-bit linear layer
            quantized_layer = bnb.nn.Linear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                compute_dtype=torch.float16,
                compress_statistics=True,
                quant_type="nf4"
            )
            
            # Copy weights and bias
            with torch.no_grad():
                quantized_layer.weight.data = module.weight.data
                if module.bias is not None:
                    quantized_layer.bias.data = module.bias.data
            
            # Replace the module
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            if parent_name:
                parent_module = model.get_submodule(parent_name)
                setattr(parent_module, child_name, quantized_layer)
            else:
                setattr(model, child_name, quantized_layer)
    
    return model


def get_trainable_params(model: nn.Module) -> Iterator[nn.Parameter]:
    """Get trainable parameters from a model (typically LoRA parameters).
    
    Args:
        model (nn.Module): Model to get trainable parameters from
        
    Yields:
        nn.Parameter: Trainable parameters
    """
    for param in model.parameters():
        if param.requires_grad:
            yield param


def count_trainable_params(model: nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters in a model.
    
    Args:
        model (nn.Module): Model to count parameters for
        
    Returns:
        Dict[str, int]: Dictionary with 'trainable' and 'total' parameter counts
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        "trainable": trainable_params,
        "total": total_params,
        "percentage": 100 * trainable_params / total_params if total_params > 0 else 0
    }


def is_qlora_available() -> bool:
    """Check if QLoRA dependencies are available.
    
    Returns:
        bool: True if QLoRA can be used, False otherwise
    """
    try:
        import peft
        import bitsandbytes
        return True
    except ImportError:
        return False


def validate_qlora_config(
    target_modules: Optional[List[str]] = None,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.1
) -> bool:
    """Validate QLoRA configuration parameters.
    
    Args:
        target_modules (List[str], optional): Target modules for LoRA
        r (int): LoRA rank
        alpha (int): LoRA alpha
        dropout (float): LoRA dropout
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    if r <= 0:
        raise ValueError(f"LoRA rank must be positive, got {r}")
    
    if alpha <= 0:
        raise ValueError(f"LoRA alpha must be positive, got {alpha}")
    
    if not 0 <= dropout <= 1:
        raise ValueError(f"LoRA dropout must be between 0 and 1, got {dropout}")
    
    if target_modules is not None and len(target_modules) == 0:
        raise ValueError("target_modules cannot be empty list")
    
    return True


def log_qlora_info(model: nn.Module, config: Dict[str, Any]) -> None:
    """Log information about QLoRA configuration and model parameters.
    
    Args:
        model (nn.Module): Model with QLoRA applied
        config (Dict[str, Any]): QLoRA configuration
    """
    param_info = count_trainable_params(model)
    
    logger.info("QLoRA Configuration:")
    logger.info(f"  - Rank (r): {config.get('r', 'N/A')}")
    logger.info(f"  - Alpha: {config.get('alpha', 'N/A')}")
    logger.info(f"  - Dropout: {config.get('dropout', 'N/A')}")
    logger.info(f"  - 4-bit quantization: {config.get('use_4bit', 'N/A')}")
    logger.info(f"  - Target modules: {config.get('target_modules', 'N/A')}")
    
    logger.info("Parameter Information:")
    logger.info(f"  - Trainable parameters: {param_info['trainable']:,}")
    logger.info(f"  - Total parameters: {param_info['total']:,}")
    logger.info(f"  - Trainable percentage: {param_info['percentage']:.2f}%")