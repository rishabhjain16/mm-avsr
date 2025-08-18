#!/usr/bin/env python3
"""Debug E2E model issue."""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datamodule.av_dataset import AVDataset
from datamodule.transforms import AudioTransform, VideoTransform, TextTransform
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E

def test_e2e_model():
    """Test E2E model step by step."""
    
    # Dataset paths
    root_dir = "/home/rishabh/Desktop/Datasets/lrs2_rf/"
    train_file = "/home/rishabh/Desktop/Datasets/lrs2_rf/labels/lrs2_train_transcript_lengths_seg16s.csv"
    
    if not os.path.exists(root_dir):
        print("Dataset not found, skipping test")
        return
    
    print("Testing E2E model step by step...")
    
    # Create dataset
    dataset = AVDataset(
        root_dir=root_dir,
        label_path=train_file,
        subset="train",
        modality="multimodal",
        audio_transform=AudioTransform("val"),
        video_transform=VideoTransform("val", vision_encoder="vit"),
        vision_encoder="vit",
        audio_encoder="whisper"
    )
    
    # Get a sample
    sample = dataset[0]
    video_data = sample["input"]
    audio_data = sample["audio_input"]
    target = sample["target"]
    
    print(f"Video data shape: {video_data.shape}")
    print(f"Audio data shape: {audio_data.shape}")
    print(f"Target length: {len(target)}")
    
    # Get vocabulary size
    text_transform = TextTransform()
    vocab_size = len(text_transform.token_list)
    
    # Create model
    model = E2E(
        odim=vocab_size,
        modality=None,
        vision_encoder="vit",
        audio_encoder="whisper",
        decoder="llm",  # Use transformer instead of llm for simpler debugging
        use_qlora=False,  # Disable QLoRA for debugging
        vision_model_name="google/vit-base-patch16-224",
        audio_model_name="openai/whisper-tiny",
        decoder_model_name="meta-llama/Llama-2-7b-hf"
    )
    model.eval()
    
    # Prepare batch data
    video_batch = video_data.unsqueeze(0)
    audio_batch = audio_data.unsqueeze(0)
    target_batch = torch.tensor(target).unsqueeze(0)
    
    video_lengths = torch.tensor([video_data.size(0)])
    audio_lengths = torch.tensor([audio_data.size(0)])
    
    print(f"Video batch shape: {video_batch.shape}")
    print(f"Audio batch shape: {audio_batch.shape}")
    print(f"Target batch shape: {target_batch.shape}")
    
    try:
        # Test individual components
        print("\n--- Testing Vision Frontend ---")
        vision_features = model.vision_frontend(video_batch)
        print(f"Vision features shape: {vision_features.shape}")
        
        print("\n--- Testing Audio Frontend ---")
        audio_features = model.audio_frontend(audio_batch)
        print(f"Audio features shape: {audio_features.shape}")
        
        print("\n--- Testing Temporal Alignment ---")
        from espnet.nets.pytorch_backend.e2e_asr_conformer import align_temporal_sequences
        vision_aligned, audio_aligned = align_temporal_sequences(vision_features, audio_features)
        print(f"Vision aligned shape: {vision_aligned.shape}")
        print(f"Audio aligned shape: {audio_aligned.shape}")
        
        print("\n--- Testing Multimodal Fusion ---")
        fused_features = model.multimodal_fusion(vision_aligned, audio_aligned)
        print(f"Fused features shape: {fused_features.shape}")
        
        print("\n--- Testing Full Forward Pass ---")
        with torch.no_grad():
            loss, loss_ctc, loss_att, acc = model(
                video_batch, video_lengths,
                target_batch, audio_batch, audio_lengths
            )
        
        print(f"✅ Full forward pass successful!")
        print(f"Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ E2E model failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_e2e_model()