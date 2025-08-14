#!/usr/bin/env python3
"""Debug ViT encoder issue."""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datamodule.av_dataset import AVDataset
from datamodule.transforms import AudioTransform, VideoTransform, TextTransform
from espnet.nets.pytorch_backend.frontend.vit_encoder import ViTEncoder

def test_vit_encoder():
    """Test ViT encoder in isolation."""
    
    # Dataset paths
    root_dir = "/home/rishabh/Desktop/Datasets/lrs2_rf/"
    train_file = "/home/rishabh/Desktop/Datasets/lrs2_rf/labels/lrs2_train_transcript_lengths_seg16s.csv"
    
    if not os.path.exists(root_dir):
        print("Dataset not found, skipping test")
        return
    
    print("Testing ViT encoder in isolation...")
    
    # Create dataset with ViT-compatible transforms
    dataset = AVDataset(
        root_dir=root_dir,
        label_path=train_file,
        subset="train",
        modality="video",
        audio_transform=AudioTransform("val"),
        video_transform=VideoTransform("val", vision_encoder="vit"),
        vision_encoder="vit",
        audio_encoder=None
    )
    
    # Get a sample
    sample = dataset[0]
    video_data = sample["input"]
    print(f"Video data shape: {video_data.shape}")
    
    # Create ViT encoder
    vit_encoder = ViTEncoder(
        model_name="google/vit-base-patch16-224",
        frozen=True,
        output_dim=768
    )
    
    # Test forward pass
    video_batch = video_data.unsqueeze(0)  # Add batch dimension
    print(f"Video batch shape: {video_batch.shape}")
    
    try:
        with torch.no_grad():
            output = vit_encoder(video_batch)
        print(f"✅ ViT encoder output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"❌ ViT encoder failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_vit_encoder()