# Requirements Document

## Introduction

This feature enhances the existing Auto-AVSR codebase to support multiple frontend encoders (vision and audio) and decoder architectures in a modular way. The goal is to enable experimentation with different model combinations (ViT, ViViT, Whisper, LLaMA, etc.) while maintaining the current directory structure and supporting QLoRA for resource-constrained training.

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to use different vision encoders (ViT, ViViT, CLIP ViT) instead of just ResNet, so that I can experiment with state-of-the-art vision models for lip-reading.

#### Acceptance Criteria

1. WHEN a user specifies a vision encoder type THEN the system SHALL load the appropriate vision model (ViT, ViViT, CLIP ViT, or ResNet)
2. WHEN using ViT-based models THEN the system SHALL support both frozen and trainable modes
3. WHEN using pretrained vision models THEN the system SHALL properly handle feature dimension alignment with the encoder
4. IF QLoRA is enabled for vision models THEN the system SHALL apply LoRA adapters to reduce memory usage

### Requirement 2

**User Story:** As a researcher, I want to use different audio encoders (Whisper, WavLM, Conformer) instead of just ResNet1D, so that I can leverage pretrained audio models for better audio-visual fusion.

#### Acceptance Criteria

1. WHEN a user specifies an audio encoder type THEN the system SHALL load the appropriate audio model (Whisper encoder, WavLM, Conformer, or ResNet1D)
2. WHEN using pretrained audio models THEN the system SHALL support frozen and fine-tunable modes
3. WHEN using Whisper encoder THEN the system SHALL extract features from the encoder layers
4. IF QLoRA is enabled for audio models THEN the system SHALL apply LoRA adapters to reduce memory usage

### Requirement 3

**User Story:** As a researcher, I want to use different decoder architectures (LLaMA, Whisper decoder, Transformer) instead of just the current Transformer decoder, so that I can experiment with large language models for sequence generation.

#### Acceptance Criteria

1. WHEN a user specifies a decoder type THEN the system SHALL load the appropriate decoder (LLaMA, Whisper decoder, or Transformer)
2. WHEN using LLaMA models THEN the system SHALL support LoRA fine-tuning while keeping the base model frozen
3. WHEN using Whisper decoder THEN the system SHALL properly handle the encoder-decoder attention mechanism
4. WHEN using any decoder THEN the system SHALL maintain compatibility with the existing CTC + attention loss training

### Requirement 4

**User Story:** As a researcher with limited GPU resources, I want to use QLoRA for large models, so that I can train with models that wouldn't normally fit in my GPU memory.

#### Acceptance Criteria

1. WHEN QLoRA is enabled THEN the system SHALL apply 4-bit quantization to the base model
2. WHEN QLoRA is enabled THEN the system SHALL add LoRA adapters for trainable parameters
3. WHEN using QLoRA THEN the system SHALL maintain training stability and convergence
4. IF a model doesn't support QLoRA THEN the system SHALL fall back to standard training with a warning

### Requirement 5

**User Story:** As a developer, I want the new modular components organized in the existing directory structure, so that the codebase remains maintainable and follows the current conventions.

#### Acceptance Criteria

1. WHEN adding new vision encoders THEN they SHALL be placed in `espnet/nets/pytorch_backend/frontend/vision/`
2. WHEN adding new audio encoders THEN they SHALL be placed in `espnet/nets/pytorch_backend/frontend/audio/`
3. WHEN adding new decoders THEN they SHALL be placed in `espnet/nets/pytorch_backend/decoder/`
4. WHEN adding LoRA utilities THEN they SHALL be placed in `espnet/nets/pytorch_backend/lora/`
5. WHEN adding model factories THEN they SHALL be placed in `espnet/nets/pytorch_backend/factories/`

### Requirement 6

**User Story:** As a researcher, I want to easily configure different model combinations through command-line arguments, so that I can quickly experiment with different architectures without code changes.

#### Acceptance Criteria

1. WHEN training THEN the system SHALL accept `--vision-encoder` argument with options (resnet, vit, vivit, clip-vit)
2. WHEN training THEN the system SHALL accept `--audio-encoder` argument with options (resnet1d, whisper, wavlm, conformer)
3. WHEN training THEN the system SHALL accept `--decoder` argument with options (transformer, llama, whisper-decoder)
4. WHEN training THEN the system SHALL accept `--use-qlora` flag to enable QLoRA training
5. WHEN both vision and audio encoders are specified THEN the system SHALL automatically use multimodal fusion
6. WHEN invalid combinations are specified THEN the system SHALL provide clear error messages

### Requirement 7

**User Story:** As a researcher, I want the system to handle feature dimension mismatches automatically, so that I can mix different encoder-decoder combinations without manual dimension calculations.

#### Acceptance Criteria

1. WHEN encoder output dimensions don't match decoder input THEN the system SHALL add appropriate projection layers
2. WHEN using different vision encoders THEN the system SHALL automatically align feature dimensions
3. WHEN using different audio encoders THEN the system SHALL automatically align feature dimensions
4. WHEN feature alignment is needed THEN the system SHALL log the dimension transformations being applied

### Requirement 8

**User Story:** As a researcher, I want to maintain backward compatibility with existing trained models, so that I can continue using my current checkpoints while experimenting with new architectures.

#### Acceptance Criteria

1. WHEN loading existing checkpoints THEN the system SHALL maintain compatibility with ResNet + Transformer models
2. WHEN using legacy model configurations THEN the system SHALL work exactly as before
3. WHEN new model types are used THEN the system SHALL clearly indicate the architecture in logs and checkpoints
4. IF checkpoint architecture doesn't match current config THEN the system SHALL provide clear error messages