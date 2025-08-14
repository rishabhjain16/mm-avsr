# Implementation Plan

- [x] 1. Set up minimal core infrastructure
  - Add new encoder files directly to existing frontend directories (resnet.py location)
  - Create simple factory functions in e2e_asr_conformer.py (no separate factory classes)
  - Add QLoRA utilities as single file in pytorch_backend directory
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 2. Extend existing E2E model with factory functions
  - Add simple factory functions inside e2e_asr_conformer.py for encoder creation
  - Extend E2E.__init__ to handle new encoder types using if/elif logic
  - Add basic validation for model combinations within E2E class
  - _Requirements: 6.1, 6.2, 6.3, 6.6_

- [x] 3. Extend argument parsing and configuration
  - Add new command-line arguments to train.py and eval.py
  - Implement automatic modality detection based on specified encoders
  - Add backward compatibility handling for existing --modality argument
  - Create validation logic for encoder-decoder combinations
  - _Requirements: 6.1, 6.2, 6.3, 6.5, 8.1, 8.2_

- [x] 4. Add vision encoders to existing frontend structure
- [x] 4.1 Create vit_encoder.py in espnet/nets/pytorch_backend/frontend/
  - Implement simple ViTEncoder class similar to existing resnet.py structure
  - Handle video input by treating frames as image patches
  - Support basic ViT variants and frozen/trainable modes
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 4.2 Create clip_encoder.py in espnet/nets/pytorch_backend/frontend/
  - Implement CLIPEncoder class following existing frontend pattern
  - Support frozen and fine-tunable modes
  - Handle feature extraction from CLIP vision encoder
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 5. Add audio encoders to existing frontend structure
- [x] 5.1 Create whisper_encoder.py in espnet/nets/pytorch_backend/frontend/
  - Implement WhisperEncoder class similar to existing resnet1d.py
  - Extract features from Whisper encoder layers
  - Support different Whisper model sizes with simple configuration
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 5.2 Create wavlm_encoder.py in espnet/nets/pytorch_backend/frontend/
  - Implement WavLMEncoder class following existing audio frontend pattern
  - Support basic WavLM variants with frozen/trainable modes
  - Handle raw audio input preprocessing
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 6. Add simple multimodal fusion to E2E model
  - Implement basic concatenation fusion directly in E2E.forward method
  - Add simple linear projection layers for dimension alignment
  - Handle temporal alignment with basic padding/truncation
  - Keep fusion logic minimal and integrated into existing E2E structure
  - _Requirements: 6.5, 7.1, 7.2, 7.3, 7.4_

- [x] 7. Add decoder options to existing decoder directory
- [x] 7.1 Create llama_decoder.py in espnet/nets/pytorch_backend/decoder/
  - Implement LLaMADecoder class following transformer_decoder.py pattern
  - Adapt LLaMA for encoder-decoder attention with visual/audio features
  - Maintain compatibility with existing beam search interface
  - _Requirements: 3.1, 3.2, 3.4_

- [x] 7.2 Create whisper_decoder.py in espnet/nets/pytorch_backend/decoder/
  - Implement WhisperDecoder class similar to existing transformer decoder
  - Support cross-attention with visual/audio encoder features
  - Ensure compatibility with existing CTC + attention loss
  - _Requirements: 3.1, 3.3, 3.4_

- [x] 8. Add simple QLoRA support
- [x] 8.1 Create qlora_utils.py in espnet/nets/pytorch_backend/
  - Implement basic QLoRA utility functions for 4-bit quantization and LoRA
  - Add simple functions to apply QLoRA to any model
  - Keep implementation minimal and focused on essential functionality
  - _Requirements: 4.1, 4.2, 4.3_

- [x] 8.2 Integrate QLoRA into E2E model initialization
  - Add QLoRA application logic directly in E2E.__init__ method
  - Apply QLoRA conditionally based on args.use_qlora flag
  - Handle parameter filtering for optimizer in lightning.py
  - _Requirements: 1.4, 2.4, 3.2, 4.4_

- [x] 9. Enhance E2E model for multimodal support
  - Modify E2E.__init__ to create vision and/or audio frontends based on args
  - Update E2E.forward to handle single or dual frontend processing
  - Add simple concatenation fusion when both frontends are present
  - Maintain backward compatibility with existing modality argument
  - _Requirements: 8.1, 8.2, 8.3_

- [x] 10. Update data loading for multimodal support
  - Modify AVDataset.__getitem__ to return both video and audio when needed
  - Update collate_pad function to handle multimodal batches
  - Keep changes minimal by extending existing data loading logic
  - _Requirements: 6.5_

- [x] 11. Add basic error handling and validation
  - Add simple validation in E2E.__init__ for model combinations
  - Add try/except blocks for missing dependencies with clear error messages
  - Keep error handling minimal and focused on common issues
  - _Requirements: 6.6, 8.4_

- [x] 12. Update training and evaluation scripts
  - Add new arguments to parse_args() in train.py and eval.py
  - Update ModelModule.__init__ to pass new args to E2E model
  - Modify eval.py to support new encoder/decoder combinations
  - Keep changes minimal by extending existing argument parsing
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 13. Test core functionality
  - Create simple test script to verify each encoder works with sample data
  - Test multimodal fusion produces correct output shapes
  - Test QLoRA reduces memory usage as expected
  - Test backward compatibility with existing ResNet + Transformer setup
  - _Requirements: 8.1, 8.2, 8.3, 8.4_