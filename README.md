# MM-AVSR: Modular Multimodal Audio-Visual Speech Recognition

## Project Overview

This repository extends the Auto-AVSR framework with a **modular architecture** that supports multiple encoder and decoder combinations for flexible audio-visual speech recognition experiments.

## Updates

`2025-01-25`: **Major Architecture Update** - Added modular encoder/decoder support with QLoRA training
- ✅ Multiple vision encoders (ResNet, ViT, ViViT, CLIP-ViT)
- ✅ Multiple audio encoders (ResNet1D, Whisper, WavLM)  
- ✅ Multiple decoders (Transformer, LLaMA, Whisper Decoder)
- ✅ Multimodal fusion with temporal alignment
- ✅ QLoRA support for memory-efficient training
- ✅ Instruction tuning for LLM decoders

`2025-01-06`: Reduced package dependencies.

`2023-07-26`: Released [real-time av-asr training code](https://github.com/pytorch/audio/tree/main/examples/avsr).

## Architecture Overview

This framework implements a **modular encoder-decoder architecture** that allows you to mix and match different components:

```
Input Modality → Frontend Encoder → Conformer → Decoder → Text Output
     ↓               ↓                ↓          ↓
   Video/Audio   ResNet/ViT/etc   Sequence   Transformer/
   Raw Data     Feature Extract   Modeling   LLaMA/Whisper
```

### Supported Configurations

| **Modality** | **Vision Encoders** | **Audio Encoders** | **Decoders** | **QLoRA** |
|--------------|--------------------|--------------------|--------------|-----------|
| Video-only | ResNet, ViT, ViViT, CLIP-ViT | - | Transformer, LLaMA, Whisper | ✅ |
| Audio-only | - | ResNet1D, Whisper, WavLM | Transformer, LLaMA, Whisper | ✅ |
| Multimodal | Any vision encoder | Any audio encoder | Any decoder | ✅ |

### Performance Benchmarks

- **Visual Speech Recognition (VSR)**: 20.3% WER on LRS3 (baseline ResNet + Transformer)
- **Audio Speech Recognition (ASR)**: 1.0% WER on LRS3 (baseline ResNet1D + Transformer)
- **Multimodal AVSR**: Improved performance with fusion of audio-visual features

## Architecture Components

### 🎥 Vision Encoders

| **Encoder** | **Type** | **Input Size** | **Output Dim** | **Use Case** |
|-------------|----------|----------------|----------------|--------------|
| **ResNet** | CNN | Variable | 512 | Lightweight, proven |
| **ViT** | Transformer | 224×224 | 768 | State-of-the-art vision |
| **ViViT** | Video Transformer | Variable | 768 | Spatio-temporal modeling |
| **CLIP-ViT** | Multimodal | 224×224 | 512 | Vision-language features |

### 🎵 Audio Encoders

| **Encoder** | **Type** | **Input** | **Output Dim** | **Use Case** |
|-------------|----------|-----------|----------------|--------------|
| **ResNet1D** | CNN | Raw waveform | 512 | Lightweight audio |
| **Whisper** | Transformer | Raw waveform | 512/768 | Speech-optimized |
| **WavLM** | Transformer | Raw waveform | 768 | Self-supervised |

### 🧠 Decoders

| **Decoder** | **Type** | **Features** | **Memory** | **Use Case** |
|-------------|----------|--------------|------------|--------------|
| **Transformer** | Standard | Fast, efficient | Low | Baseline performance |
| **LLaMA** | LLM | Instruction tuning | High* | Advanced reasoning |
| **Whisper Decoder** | Speech LLM | Speech-optimized | Medium* | Speech-specific |

*\*With QLoRA support for memory efficiency*

### 🔗 Multimodal Fusion

- **Temporal Alignment**: Automatic alignment of audio-visual sequences
- **Fusion Types**: Concatenation, Addition, Gated fusion
- **Output**: 768-dim unified representation for Conformer processing

## Setup

### 1. Install Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio pytorch-lightning sentencepiece av wandb soundfile

# For advanced encoders and LLM decoders
pip install transformers timm

# For QLoRA memory optimization
pip install peft bitsandbytes

# For CLIP encoder (if using CLIP-ViT)
pip install torch-vision
```

### 2. Dataset Preparation

Prepare your dataset following the [preparation guide](./preparation).

## Training

### Basic Training Commands

#### 1. Video-Only Speech Recognition (Lip Reading)

```bash
# ResNet + Transformer (baseline)
python train.py --exp-dir ./exp --exp-name vsr_resnet_transformer \
                --vision-encoder resnet --decoder transformer \
                --root-dir /path/to/dataset --train-file train.csv \
                --num-nodes 1 --gpus 1

# ViT + LLaMA with QLoRA (advanced)
python train.py --exp-dir ./exp --exp-name vsr_vit_llama \
                --vision-encoder vit --decoder llm \
                --vision-model-name google/vit-base-patch16-224 \
                --decoder-model-name meta-llama/Llama-2-7b-hf \
                --use-qlora --qlora-r 16 --qlora-alpha 32 \
                --root-dir /path/to/dataset --train-file train.csv \
                --num-nodes 1 --gpus 1
```

#### 2. Audio-Only Speech Recognition

```bash
# Whisper + Transformer
python train.py --exp-dir ./exp --exp-name asr_whisper_transformer \
                --audio-encoder whisper --decoder transformer \
                --audio-model-name openai/whisper-base \
                --root-dir /path/to/dataset --train-file train.csv \
                --num-nodes 1 --gpus 1

# WavLM + LLaMA with QLoRA
python train.py --exp-dir ./exp --exp-name asr_wavlm_llama \
                --audio-encoder wavlm --decoder llm \
                --audio-model-name microsoft/wavlm-base \
                --decoder-model-name meta-llama/Llama-2-7b-hf \
                --use-qlora \
                --root-dir /path/to/dataset --train-file train.csv \
                --num-nodes 1 --gpus 1
```

#### 3. Multimodal Audio-Visual Speech Recognition

```bash
# ResNet + ResNet1D + Transformer (baseline multimodal)
python train.py --exp-dir ./exp --exp-name avsr_resnet_transformer \
                --vision-encoder resnet --audio-encoder resnet1d \
                --decoder transformer --fusion-type concat \
                --root-dir /path/to/dataset --train-file train.csv \
                --num-nodes 1 --gpus 1

# ViT + Whisper + LLaMA with QLoRA (advanced multimodal)
python train.py --exp-dir ./exp --exp-name avsr_vit_whisper_llama \
                --vision-encoder vit --audio-encoder whisper \
                --decoder llm --fusion-type gated \
                --vision-model-name google/vit-base-patch16-224 \
                --audio-model-name openai/whisper-base \
                --decoder-model-name meta-llama/Llama-2-7b-hf \
                --use-qlora --qlora-r 16 --qlora-alpha 32 \
                --root-dir /path/to/dataset --train-file train.csv \
                --num-nodes 1 --gpus 1
```

### Advanced Configuration Options

<details>
  <summary><strong>Encoder/Decoder Arguments</strong></summary>

**Vision Encoders:**
- `--vision-encoder`: `resnet`, `vit`, `vivit`, `clip-vit`
- `--vision-model-name`: Specific HuggingFace model (e.g., `google/vit-base-patch16-224`)
- `--unfreeze-vision`: Unfreeze vision encoder for full training

**Audio Encoders:**
- `--audio-encoder`: `resnet1d`, `whisper`, `wavlm`
- `--audio-model-name`: Specific HuggingFace model (e.g., `openai/whisper-base`)
- `--unfreeze-audio`: Unfreeze audio encoder for full training

**Decoders:**
- `--decoder`: `transformer`, `llm`, `whisper-decoder`
- `--decoder-model-name`: Specific model (e.g., `meta-llama/Llama-2-7b-hf`)

**Multimodal Fusion:**
- `--fusion-type`: `concat`, `add`, `gated`

</details>

<details>
  <summary><strong>QLoRA Memory Optimization</strong></summary>

- `--use-qlora`: Enable QLoRA for memory-efficient training
- `--qlora-r`: LoRA rank (default: 16, lower = less memory)
- `--qlora-alpha`: LoRA alpha (default: 32)

**Memory Usage Guide:**
- **Without QLoRA**: LLaMA-7B requires ~28GB VRAM
- **With QLoRA**: LLaMA-7B requires ~8-12GB VRAM
- **Recommended**: Use QLoRA for any LLM decoder

</details>

<details>
  <summary><strong>Standard Training Arguments</strong></summary>

**Required:**
- `--exp-dir`: Experiment directory (default: `./exp`)
- `--exp-name`: Experiment name
- `--root-dir`: Dataset root directory
- `--train-file`: Training file list
- `--num-nodes`: Number of machines (default: 1)

**Optional:**
- `--gpus`: GPUs per machine (default: 1)
- `--max-epochs`: Training epochs (default: 10)
- `--lr`: Learning rate (default: 1e-3)
- `--max-frames`: Max frames per batch (default: 2000)
- `--ctc-weight`: CTC loss weight (default: 0.1)
- `--pretrained-model-path`: Path to pretrained model
- `--val-file`: Validation file (default: `lrs2_val_transcript_lengths_seg16s.csv`)
- `--test-file`: Test file (default: `lrs2_test_transcript_lengths_seg16s.csv`)

</details>

### Legacy Support

The framework maintains backward compatibility with the original Auto-AVSR arguments:

```bash
# Legacy modality argument (deprecated but supported)
python train.py --modality video --exp-dir ./exp --exp-name legacy_vsr \
                --root-dir /path/to/dataset --train-file train.csv --num-nodes 1
```

## Evaluation

### Basic Evaluation Commands

```bash
# Evaluate video-only model
python eval.py --vision-encoder vit --decoder llm \
               --vision-model-name google/vit-base-patch16-224 \
               --decoder-model-name meta-llama/Llama-2-7b-hf \
               --pretrained-model-path ./exp/vsr_vit_llama/model_avg_10.pth \
               --root-dir /path/to/dataset --test-file test.csv

# Evaluate audio-only model  
python eval.py --audio-encoder whisper --decoder transformer \
               --audio-model-name openai/whisper-base \
               --pretrained-model-path ./exp/asr_whisper_transformer/model_avg_10.pth \
               --root-dir /path/to/dataset --test-file test.csv

# Evaluate multimodal model
python eval.py --vision-encoder vit --audio-encoder whisper --decoder llm \
               --vision-model-name google/vit-base-patch16-224 \
               --audio-model-name openai/whisper-base \
               --decoder-model-name meta-llama/Llama-2-7b-hf \
               --fusion-type gated \
               --pretrained-model-path ./exp/avsr_vit_whisper_llama/model_avg_10.pth \
               --root-dir /path/to/dataset --test-file test.csv
```

### Legacy Evaluation (Backward Compatible)

```bash
# Original Auto-AVSR evaluation format
python eval.py --modality video \
               --root-dir /path/to/dataset \
               --test-file test.csv \
               --pretrained-model-path ./exp/model.pth
```

<details>
  <summary><strong>Evaluation Arguments</strong></summary>

**Model Configuration (must match training):**
- `--vision-encoder`, `--audio-encoder`, `--decoder`: Same as training
- `--vision-model-name`, `--audio-model-name`, `--decoder-model-name`: Same as training
- `--fusion-type`: Same as training (for multimodal)

**Required:**
- `--pretrained-model-path`: Path to trained model checkpoint
- `--root-dir`: Dataset root directory
- `--test-file`: Test file list

**Optional:**
- `--decode-snr-target`: SNR level (default: 999999)
- `--output-json`: Save results to JSON file
- `--debug`: Enable debug logging

</details>

## Model Zoo & Performance

### Baseline Models (Original Auto-AVSR)

<details open>
<summary><strong>LRS3 Benchmark Results</strong></summary>

| **Model** | **Architecture** | **Training Data (h)** | **WER (%)** | **Params (M)** | **Download** |
|-----------|------------------|:---------------------:|:-----------:|:--------------:|:------------:|
| VSR Baseline | ResNet + Transformer | 438 | 36.0 | 250 | [Link](https://drive.google.com/file/d/12PNM5szUsk_CuaV1yB9dL_YWvSM1zvAd/view?usp=sharing) |
| VSR Large | ResNet + Transformer | 1759 | 24.6 | 250 | [Link](https://drive.google.com/file/d/1shcWXUK2iauRhW9NbwCc25FjU1CoMm8i/view?usp=sharing) |
| VSR SOTA | ResNet + Transformer | 3291 | **20.3** | 250 | [Link](https://drive.google.com/file/d/1r1kx7l9sWnDOCnaFHIGvOtzuhFyFA88_/view?usp=sharing) |
| ASR Baseline | ResNet1D + Transformer | 438 | 2.0 | 243 | [Link](https://drive.google.com/file/d/1IBMkI7XyZo8mF3rz109rXrMH7MyxRuiY/view?usp=sharing) |
| ASR Large | ResNet1D + Transformer | 1759 | **1.0** | 243 | [Link](https://drive.google.com/file/d/1YN9lwZN6iWn2qNQRpfpGpnf2r6ZTQqVT/view?usp=sharing) |

</details>

### Modular Architecture Performance

<details>
<summary><strong>Expected Performance by Configuration</strong></summary>

| **Modality** | **Encoder** | **Decoder** | **Expected WER** | **Memory (GB)** | **Training Time** |
|--------------|-------------|-------------|:----------------:|:---------------:|:-----------------:|
| **Video-only** | ResNet + Transformer | Baseline | 20-25% | 4-6 | 1x |
| | ViT + Transformer | Improved | 18-23% | 8-12 | 1.5x |
| | ViT + LLaMA (QLoRA) | Advanced | 15-20% | 8-12 | 2x |
| **Audio-only** | Whisper + Transformer | Strong | 1-2% | 6-8 | 1.2x |
| | WavLM + LLaMA (QLoRA) | Advanced | 0.8-1.5% | 8-12 | 2x |
| **Multimodal** | ResNet + ResNet1D + Transformer | Baseline | 15-20% | 6-10 | 1.5x |
| | ViT + Whisper + LLaMA (QLoRA) | SOTA | **10-15%** | 12-16 | 3x |

*Performance estimates based on similar architectures and datasets*

</details>

### Model Compatibility

- ✅ **Backward Compatible**: All original Auto-AVSR models work with new framework
- ✅ **Cross-Architecture**: Can mix encoders/decoders from different model families
- ✅ **Transfer Learning**: Pretrained models can be used as feature extractors
- ✅ **QLoRA Support**: Memory-efficient training for large models


## Quick Start Examples

### 1. Simple Video-Only Training

```bash
# Train a basic lip reading model
python train.py \
    --exp-dir ./exp --exp-name my_first_vsr \
    --vision-encoder resnet --decoder transformer \
    --root-dir /path/to/lrs3 --train-file train.csv \
    --num-nodes 1 --gpus 1 --max-epochs 20
```

### 2. Advanced Multimodal with QLoRA

```bash
# Train state-of-the-art multimodal model with memory optimization
python train.py \
    --exp-dir ./exp --exp-name advanced_avsr \
    --vision-encoder vit --audio-encoder whisper --decoder llm \
    --vision-model-name google/vit-base-patch16-224 \
    --audio-model-name openai/whisper-base \
    --decoder-model-name meta-llama/Llama-2-7b-hf \
    --fusion-type gated --use-qlora \
    --root-dir /path/to/lrs3 --train-file train.csv \
    --num-nodes 1 --gpus 1 --max-epochs 30 --lr 5e-5
```

### 3. Evaluate Trained Model

```bash
# Evaluate the trained model
python eval.py \
    --vision-encoder vit --audio-encoder whisper --decoder llm \
    --vision-model-name google/vit-base-patch16-224 \
    --audio-model-name openai/whisper-base \
    --decoder-model-name meta-llama/Llama-2-7b-hf \
    --fusion-type gated \
    --pretrained-model-path ./exp/advanced_avsr/model_avg_10.pth \
    --root-dir /path/to/lrs3 --test-file test.csv
```

## Tutorials & Documentation

- [x] [Cropping Mouth from Video](./tutorials/mouth_cropping.ipynb)
- [x] [Audio/Visual Speech Recognition](./tutorials/inference.ipynb)  
- [x] [Feature Extraction](./tutorials/feature_extraction.ipynb)
- [x] [Modular Architecture Guide](./understand.md)
- [ ] Advanced QLoRA Training (Coming Soon)
- [ ] Custom Encoder Integration (Coming Soon)
- [ ] Multimodal Fusion Strategies (Coming Soon)

## Troubleshooting

### Common Issues

<details>
<summary><strong>Memory Issues</strong></summary>

**Problem**: Out of memory errors during training

**Solutions**:
1. **Enable QLoRA**: Add `--use-qlora` for LLM decoders
2. **Reduce batch size**: Lower `--max-frames` (try 1000, 800, 600)
3. **Use smaller models**: Try `openai/whisper-tiny` instead of `whisper-base`
4. **Gradient checkpointing**: Automatically enabled for long sequences

</details>

<details>
<summary><strong>Model Loading Issues</strong></summary>

**Problem**: "Model not found" or "Incompatible checkpoint"

**Solutions**:
1. **Check model names**: Ensure HuggingFace model names are correct
2. **Install dependencies**: `pip install transformers peft bitsandbytes`
3. **Match training config**: Evaluation args must match training configuration
4. **Check checkpoint path**: Verify the `.pth` file exists

</details>

<details>
<summary><strong>Performance Issues</strong></summary>

**Problem**: Poor WER or slow convergence

**Solutions**:
1. **Adjust learning rate**: Try 1e-4 for pretrained models, 1e-3 for from scratch
2. **Increase epochs**: LLM decoders may need 30-50 epochs
3. **Check data quality**: Verify dataset preprocessing
4. **Try different fusion**: Experiment with `--fusion-type gated` for multimodal

</details>

## Citation

If you find this repository helpful, please consider citing the original Auto-AVSR work:

```bibtex
@inproceedings{ma2023auto,
  author={Ma, Pingchuan and Haliassos, Alexandros and Fernandez-Lopez, Adriana and Chen, Honglie and Petridis, Stavros and Pantic, Maja},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels},
  year={2023},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10096889}
}
```

For the modular architecture extensions, please also consider citing relevant encoder/decoder papers:
- **ViT**: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- **LLaMA**: Touvron et al., "LLaMA: Open and Efficient Foundation Language Models"  
- **Whisper**: Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision"

## Architecture Details

### Data Flow

```
Input → Frontend Encoder → Conformer → Decoder → Output
  ↓         ↓               ↓          ↓        ↓
Raw A/V   Feature        Sequence   Language   Text
Data      Extraction     Modeling   Modeling   Tokens
```

**Single Modality Flow:**
```
Video/Audio → Encoder → proj_encoder (MLP) → Conformer → Decoder
(B,T,raw)   → (B,T,dim) → (B,T,768)        → (B,T,768) → (B,T,vocab)
```

**Multimodal Flow:**
```
Video → Vision Encoder → (B,T,v_dim) ↘
                                      Fusion (MLP) → Conformer → Decoder  
Audio → Audio Encoder  → (B,T,a_dim) ↗   (B,T,768)   → (B,T,768) → (B,T,vocab)
```

### Key Features

- **Modular Design**: Mix and match any encoder with any decoder
- **Automatic Modality Detection**: Framework detects modality from specified encoders
- **Temporal Alignment**: Automatic alignment of audio-visual sequences
- **Memory Optimization**: QLoRA support for large models
- **Instruction Tuning**: LLM decoders support task instructions
- **Backward Compatibility**: Works with existing Auto-AVSR models

## Acknowledgements

This repository extends the original [Auto-AVSR](https://github.com/mpc001/auto_avsr) framework and builds upon:

- **Core Framework**: [ESPnet](https://github.com/espnet/espnet), [PyTorch Lightning](https://github.com/Lightning-AI/lightning)
- **Audio Processing**: [torchaudio](https://github.com/pytorch/audio), [Whisper](https://github.com/openai/whisper)
- **Vision Models**: [timm](https://github.com/huggingface/pytorch-image-models), [transformers](https://github.com/huggingface/transformers)
- **Memory Optimization**: [PEFT](https://github.com/huggingface/peft), [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- **Original Research**: [Auto-AVSR](https://github.com/mpc001/auto_avsr), [AV-HuBERT](https://github.com/facebookresearch/av_hubert)

## License

- **Code**: Apache 2.0 License
- **Pretrained Models**: May have individual licenses based on training data
- **Dependencies**: Subject to their respective licenses (MIT, Apache 2.0, etc.)

## Contributing

Contributions are welcome! Please feel free to:

- 🐛 Report bugs or issues
- 💡 Suggest new features or encoders
- 🔧 Submit pull requests
- 📖 Improve documentation
- 🧪 Share experimental results

For major changes, please open an issue first to discuss the proposed changes.

## Contact

**Original Auto-AVSR**: [Pingchuan Ma](mailto:mapingchuan0420@gmail.com)

**Modular Extensions**: Feel free to open GitHub issues for questions or discussions about the modular architecture features.
