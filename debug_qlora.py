#!/usr/bin/env python3

import torch
import logging
logging.basicConfig(level=logging.INFO)

# Test QLoRA loading directly
try:
    from transformers import LlamaModel, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    
    print("Testing QLoRA loading...")
    
    model_name = "meta-llama/Llama-3.2-1B"
    
    # Create quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        torch_dtype=torch.float16
    )
    
    print("Loading model with quantization...")
    model = LlamaModel.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Model loaded successfully!")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Check parameter dtypes
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}")
        if "uint8" in str(param.dtype):
            print(f"Found uint8 parameter: {name}")
        break  # Just check first few
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    
    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    print("LoRA applied successfully!")
    
    # Test moving to device
    print("Testing device movement...")
    model = model.cuda()
    print("Device movement successful!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()