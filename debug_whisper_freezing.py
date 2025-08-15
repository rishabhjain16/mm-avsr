#!/usr/bin/env python3

"""
Debug script to check if Whisper encoder is properly frozen.
"""

import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_whisper_freezing():
    """Test if Whisper encoder parameters are properly frozen."""
    
    try:
        from espnet.nets.pytorch_backend.frontend.whisper_encoder import WhisperEncoder
        
        print("Testing Whisper encoder freezing...")
        
        # Test with freeze_encoder=True (default)
        print("\n1. Testing with freeze_encoder=True:")
        encoder_frozen = WhisperEncoder(
            model_name="openai/whisper-base",
            freeze_encoder=True
        )
        
        frozen_params = 0
        trainable_params = 0
        
        for name, param in encoder_frozen.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"  TRAINABLE: {name} - {param.numel()} params")
            else:
                frozen_params += param.numel()
        
        print(f"  Frozen parameters: {frozen_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Total parameters: {frozen_params + trainable_params:,}")
        
        # Test with freeze_encoder=False
        print("\n2. Testing with freeze_encoder=False:")
        encoder_unfrozen = WhisperEncoder(
            model_name="openai/whisper-base",
            freeze_encoder=False
        )
        
        frozen_params_2 = 0
        trainable_params_2 = 0
        
        for name, param in encoder_unfrozen.named_parameters():
            if param.requires_grad:
                trainable_params_2 += param.numel()
            else:
                frozen_params_2 += param.numel()
        
        print(f"  Frozen parameters: {frozen_params_2:,}")
        print(f"  Trainable parameters: {trainable_params_2:,}")
        print(f"  Total parameters: {frozen_params_2 + trainable_params_2:,}")
        
        # Verify the difference
        print(f"\n3. Verification:")
        print(f"  Difference in trainable params: {trainable_params_2 - trainable_params:,}")
        
        if trainable_params == 0 and trainable_params_2 > 0:
            print("  ✅ Freezing is working correctly!")
        else:
            print("  ❌ Freezing might not be working properly!")
            
        return trainable_params == 0 and trainable_params_2 > 0
        
    except Exception as e:
        print(f"Error testing Whisper freezing: {e}")
        return False

def test_memory_usage():
    """Test memory usage with frozen vs unfrozen encoder."""
    
    try:
        from espnet.nets.pytorch_backend.frontend.whisper_encoder import WhisperEncoder
        import torch
        
        print("\n4. Testing memory usage:")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            print(f"  Initial GPU memory: {initial_memory / 1024**2:.1f} MB")
        
        # Test frozen encoder
        encoder_frozen = WhisperEncoder(
            model_name="openai/whisper-base",
            freeze_encoder=True
        )
        
        if torch.cuda.is_available():
            encoder_frozen = encoder_frozen.cuda()
            frozen_memory = torch.cuda.memory_allocated()
            print(f"  Memory with frozen encoder: {frozen_memory / 1024**2:.1f} MB")
        
        # Create dummy input
        batch_size = 2
        seq_len = 1000
        dummy_input = torch.randn(batch_size, seq_len, 1)
        
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
        
        # Forward pass with frozen encoder
        with torch.no_grad():
            output_frozen = encoder_frozen(dummy_input)
            
        if torch.cuda.is_available():
            after_forward_memory = torch.cuda.memory_allocated()
            print(f"  Memory after forward (frozen): {after_forward_memory / 1024**2:.1f} MB")
        
        print(f"  Output shape (frozen): {output_frozen.shape}")
        
        # Clean up
        del encoder_frozen, output_frozen, dummy_input
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error testing memory usage: {e}")

if __name__ == "__main__":
    print("Whisper Encoder Freezing Debug")
    print("=" * 40)
    
    success = test_whisper_freezing()
    test_memory_usage()
    
    if success:
        print("\n✅ Whisper encoder freezing appears to be working correctly.")
        print("The memory issue might be elsewhere in the pipeline.")
    else:
        print("\n❌ There might be an issue with Whisper encoder freezing.")