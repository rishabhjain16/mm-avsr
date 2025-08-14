# ViViT (Video Vision Transformer) Encoder

This directory contains the implementation of ViViT (Video Vision Transformer) encoder for video understanding tasks, specifically designed for lip-reading and audio-visual speech recognition.

## Overview

ViViT is a Video Vision Transformer that extends the Vision Transformer (ViT) architecture to handle video data with native temporal modeling capabilities. This implementation uses the official ViViT model from HuggingFace transformers library, providing state-of-the-art video understanding for lip-reading tasks.

## Features

- **Official ViViT Implementation**: Uses the real ViViT model from transformers library
- **Temporal Video Understanding**: Native support for video sequences with temporal attention
- **Flexible Frame Handling**: Automatically adjusts input frame counts to match model expectations
- **Multiple Input Formats**: Supports both 4D and 5D input tensors
- **Pre-trained Models**: Leverages pre-trained ViViT models from HuggingFace
- **Configurable Temporal Pooling**: Different strategies for temporal feature aggregation

## Usage

### Basic Usage

```python
from espnet.nets.pytorch_backend.frontend.vivit_encoder import vivit_encoder

# Create ViViT encoder with default settings
encoder = vivit_encoder()

# Process video input (B, T, H, W) or (B, C, T, H, W)
video_input = torch.randn(2, 16, 224, 224)  # Batch=2, Frames=16, Height=224, Width=224
output = encoder(video_input)  # Output shape: (2, 8, 768)
```

### Advanced Configuration

```python
# ViViT with custom configuration
encoder = vivit_encoder(
    model_name="google/vivit-b-16x2-kinetics400",  # ViViT model
    frozen=False,                                  # Allow fine-tuning
    output_dim=512,                               # Custom output dimension
    num_frames=32,                                # Expected number of frames
    temporal_pooling="mean",                      # "mean", "max", or "cls"
)
```

### Integration with E2E Model

```python
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E

# Single modality (video-only)
model = E2E(
    odim=5000,
    vision_encoder='vivit',
    vision_model_name='google/vivit-b-16x2-kinetics400'
)

# Multimodal (audio-visual)
model = E2E(
    odim=5000,
    modality='multimodal',
    vision_encoder='vivit',
    audio_encoder='resnet1d',
    vision_model_name='google/vivit-b-16x2-kinetics400'
)
```

## Architecture Details

### ViViT Model
- Uses official ViViT implementation from HuggingFace transformers
- Factorized spatial-temporal attention for efficient video processing
- Tubelet embeddings for joint spatial-temporal tokenization
- Pre-trained on Kinetics-400 dataset for robust video understanding

### Temporal Processing
- Automatically handles variable frame lengths through interpolation
- Supports frame sampling for longer sequences
- Native temporal attention mechanisms
- Configurable temporal pooling strategies

### Frame Adjustment
- Input frames are automatically adjusted to match model expectations
- Interpolation for shorter sequences
- Uniform sampling for longer sequences
- Default: expects 32 frames for optimal performance

## Input Formats

The encoder supports multiple input formats:

1. **4D Input**: `(B, T, H, W)` - Grayscale video, automatically converted to RGB
2. **5D Input**: `(B, C, T, H, W)` - RGB video (C=3) or grayscale (C=1, converted to RGB)

## Output

- **Shape**: `(B, T_out, output_dim)`
- **T_out**: Temporal resolution after patching (typically T_input // temporal_patch_size)
- **output_dim**: Configurable output dimension (default: 768)

## Model Variants

Supported ViViT models:
- `google/vivit-b-16x2-kinetics400` (default) - Base model, 16x16 spatial patches, 2-frame tubelets
- `google/vivit-l-16x2-kinetics400` - Large model variant
- `google/vivit-h-14x2-kinetics400` - Huge model variant
- Any compatible ViViT model from HuggingFace

## Performance Considerations

- **Memory Usage**: ViViT requires significant GPU memory due to attention mechanisms
- **Temporal Length**: Longer sequences require more memory and computation
- **Attention Type**: Joint attention is more expensive than factorized
- **QLoRA Support**: Can be used with QLoRA for memory-efficient training

## Training Tips

1. **Fine-tuning**: Set `frozen=False` for better performance on lip-reading tasks
2. **Learning Rate**: Use lower learning rates for pre-trained components
3. **Temporal Augmentation**: Consider temporal cropping and frame sampling
4. **Mixed Precision**: Use automatic mixed precision for memory efficiency

## References

- Arnab, A., et al. "ViViT: A Video Vision Transformer." ICCV 2021.
- Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.