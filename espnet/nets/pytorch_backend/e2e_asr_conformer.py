#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import torch
import torch.nn.functional as F
import logging

from espnet.nets.pytorch_backend.frontend.resnet import video_resnet
from espnet.nets.pytorch_backend.frontend.resnet1d import audio_resnet
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.encoder.conformer_encoder import ConformerEncoder
from espnet.nets.pytorch_backend.decoder.transformer_decoder import TransformerDecoder
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.scorers.ctc import CTCPrefixScorer


def create_vision_encoder(encoder_type="resnet", model_name=None, **kwargs):
    """Factory function to create vision encoders.
    
    Args:
        encoder_type (str): Type of vision encoder ('resnet', 'vit', 'vivit', 'clip-vit')
        model_name (str): Specific model name for the encoder
        **kwargs: Additional arguments for encoder initialization
        
    Returns:
        torch.nn.Module: Vision encoder instance
    """
    try:
        if encoder_type == "resnet":
            return video_resnet()
        elif encoder_type == "vit":
            try:
                from espnet.nets.pytorch_backend.frontend.vit_encoder import vit_encoder
                return vit_encoder(model_name=model_name, **kwargs)
            except ImportError as e:
                raise ImportError(f"ViT encoder dependencies not available. "
                                f"Please install required packages: transformers, timm. Error: {e}")
        elif encoder_type == "vivit":
            try:
                from espnet.nets.pytorch_backend.frontend.vivit_encoder import vivit_encoder
                return vivit_encoder(model_name=model_name, **kwargs)
            except ImportError as e:
                raise ImportError(f"ViViT encoder dependencies not available. "
                                f"Please install required packages: transformers. Error: {e}")
        elif encoder_type == "clip-vit":
            try:
                from espnet.nets.pytorch_backend.frontend.clip_encoder import clip_vit_encoder
                return clip_vit_encoder(model_name=model_name, **kwargs)
            except ImportError as e:
                raise ImportError(f"CLIP encoder dependencies not available. "
                                f"Please install required packages: transformers, torch-vision. Error: {e}")
        else:
            raise ValueError(f"Unknown vision encoder type: {encoder_type}. "
                           f"Valid options: resnet, vit, vivit, clip-vit")
    except Exception as e:
        logging.error(f"Failed to create vision encoder '{encoder_type}': {e}")
        raise


def create_audio_encoder(encoder_type="resnet1d", model_name=None, **kwargs):
    """Factory function to create audio encoders.
    
    Args:
        encoder_type (str): Type of audio encoder ('resnet1d', 'whisper', 'wavlm', 'conformer')
        model_name (str): Specific model name for the encoder
        **kwargs: Additional arguments for encoder initialization
        
    Returns:
        torch.nn.Module: Audio encoder instance
    """
    try:
        if encoder_type == "resnet1d":
            return audio_resnet()
        elif encoder_type == "whisper":
            try:
                from espnet.nets.pytorch_backend.frontend.whisper_encoder import audio_whisper_encoder
                return audio_whisper_encoder(model_name=model_name, **kwargs)
            except ImportError as e:
                raise ImportError(f"Whisper encoder dependencies not available. "
                                f"Please install required packages: transformers, torch-audio. Error: {e}")
        elif encoder_type == "wavlm":
            try:
                from espnet.nets.pytorch_backend.frontend.wavlm_encoder import audio_wavlm_encoder
                return audio_wavlm_encoder(model_name=model_name, **kwargs)
            except ImportError as e:
                raise ImportError(f"WavLM encoder dependencies not available. "
                                f"Please install required packages: transformers, torch-audio. Error: {e}")
        elif encoder_type == "conformer":
            # Placeholder for Conformer encoder - will be implemented in later tasks
            raise NotImplementedError("Conformer encoder will be implemented in later tasks")
        else:
            raise ValueError(f"Unknown audio encoder type: {encoder_type}. "
                           f"Valid options: resnet1d, whisper, wavlm, conformer")
    except Exception as e:
        logging.error(f"Failed to create audio encoder '{encoder_type}': {e}")
        raise


def create_decoder(decoder_type="transformer", odim=None, model_name=None, **kwargs):
    """Factory function to create decoders.
    
    Args:
        decoder_type (str): Type of decoder ('transformer', 'llama', 'whisper-decoder')
        odim (int): Output dimension
        model_name (str): Specific model name for the decoder
        **kwargs: Additional arguments for decoder initialization
        
    Returns:
        torch.nn.Module: Decoder instance
    """
    try:
        if decoder_type == "transformer":
            if odim is None:
                raise ValueError("odim (output dimension) is required for transformer decoder")
            return TransformerDecoder(
                odim=odim,
                attention_dim=768,
                attention_heads=12,
                linear_units=3072,
                num_blocks=6,
            )
        elif decoder_type == "llama":
            try:
                from espnet.nets.pytorch_backend.decoder.llama_decoder import LLaMADecoder
                if odim is None:
                    raise ValueError("odim (output dimension) is required for LLaMA decoder")
                # Use default model if none specified
                if model_name is None:
                    model_name = "meta-llama/Llama-2-7b-hf"
                return LLaMADecoder(odim=odim, model_name=model_name, **kwargs)
            except ImportError as e:
                raise ImportError(f"LLaMA decoder dependencies not available. "
                                f"Please install required packages: transformers, peft, bitsandbytes. Error: {e}")
        elif decoder_type == "whisper-decoder":
            try:
                from espnet.nets.pytorch_backend.decoder.whisper_decoder import WhisperDecoder
                if odim is None:
                    raise ValueError("odim (output dimension) is required for Whisper decoder")
                return WhisperDecoder(odim=odim, model_name=model_name, **kwargs)
            except ImportError as e:
                raise ImportError(f"Whisper decoder dependencies not available. "
                                f"Please install required packages: transformers, torch-audio. Error: {e}")
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}. "
                           f"Valid options: transformer, llama, whisper-decoder")
    except Exception as e:
        logging.error(f"Failed to create decoder '{decoder_type}': {e}")
        raise


def get_encoder_output_dim(encoder, encoder_type, model_name=None):
    """Get the output dimension of an encoder.
    
    Args:
        encoder: The encoder instance
        encoder_type (str): Type of encoder
        model_name (str): Model name for dimension lookup
        
    Returns:
        int: Output dimension of the encoder
    """
    # Try to get dimension from encoder if it has the method
    if hasattr(encoder, 'get_output_dim'):
        return encoder.get_output_dim()
    
    # Default dimensions for known encoder types
    if encoder_type == "resnet":
        return 512
    elif encoder_type == "resnet1d":
        return 512
    elif encoder_type == "vit":
        # ViT dimensions vary by model
        if model_name and "large" in model_name:
            return 1024
        else:
            return 768  # Default for base models
    elif encoder_type == "vivit":
        # ViViT dimensions (same as base ViT since it uses ViT backbone)
        if model_name and "large" in model_name:
            return 1024
        else:
            return 768  # Default for base models
    elif encoder_type == "clip-vit":
        # CLIP dimensions vary by model
        if model_name and "large" in model_name:
            return 768
        else:
            return 512  # Default for base models
    elif encoder_type == "whisper":
        # Whisper dimensions vary by model size
        if model_name:
            if "large" in model_name:
                return 1280
            elif "medium" in model_name:
                return 1024
            elif "small" in model_name:
                return 768
            elif "base" in model_name:
                return 512
            elif "tiny" in model_name:
                return 384
            else:
                return 512  # Default fallback
        return 512
    elif encoder_type == "wavlm":
        # WavLM dimensions
        if model_name and "large" in model_name:
            return 1024
        else:
            return 768  # Default for base models
    else:
        # Fallback: try to infer from a dummy forward pass
        try:
            with torch.no_grad():
                if encoder_type in ["resnet", "vit", "vivit", "clip-vit"]:
                    # Vision encoder - dummy video input
                    dummy_input = torch.randn(1, 10, 96, 96)  # B, T, H, W
                else:
                    # Audio encoder - dummy audio input
                    dummy_input = torch.randn(1, 1000, 1)  # B, T, C
                
                output = encoder(dummy_input)
                return output.size(-1)
        except Exception as e:
            logging.warning(f"Could not determine output dimension for {encoder_type}: {e}")
            return 512  # Safe fallback


def align_temporal_sequences(vision_features, audio_features, method="pad_truncate"):
    """Align temporal dimensions of vision and audio features.
    
    Args:
        vision_features (torch.Tensor): Vision features (B, T_v, D_v)
        audio_features (torch.Tensor): Audio features (B, T_a, D_a)
        method (str): Alignment method ("pad_truncate", "interpolate")
        
    Returns:
        tuple: Aligned (vision_features, audio_features)
    """
    B_v, T_v, D_v = vision_features.shape
    B_a, T_a, D_a = audio_features.shape
    
    assert B_v == B_a, f"Batch sizes must match: {B_v} vs {B_a}"
    
    if T_v == T_a:
        return vision_features, audio_features
    
    if method == "pad_truncate":
        # Use the shorter sequence length and pad/truncate accordingly
        T_target = min(T_v, T_a)
        
        # Truncate if necessary
        vision_aligned = vision_features[:, :T_target, :]
        audio_aligned = audio_features[:, :T_target, :]
        
        logging.info(f"Temporal alignment: vision {T_v} -> {T_target}, audio {T_a} -> {T_target}")
        
    elif method == "interpolate":
        # Interpolate to match the longer sequence
        T_target = max(T_v, T_a)
        
        if T_v != T_target:
            # Interpolate vision features
            vision_features_transposed = vision_features.transpose(1, 2)  # B, D_v, T_v
            vision_aligned = F.interpolate(
                vision_features_transposed, 
                size=T_target, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)  # B, T_target, D_v
        else:
            vision_aligned = vision_features
            
        if T_a != T_target:
            # Interpolate audio features
            audio_features_transposed = audio_features.transpose(1, 2)  # B, D_a, T_a
            audio_aligned = F.interpolate(
                audio_features_transposed, 
                size=T_target, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)  # B, T_target, D_a
        else:
            audio_aligned = audio_features
            
        logging.info(f"Temporal alignment (interpolate): vision {T_v} -> {T_target}, audio {T_a} -> {T_target}")
    else:
        raise ValueError(f"Unknown alignment method: {method}")
    
    return vision_aligned, audio_aligned


class MultimodalFusion(torch.nn.Module):
    """Simple multimodal fusion module for combining vision and audio features."""
    
    def __init__(self, vision_dim, audio_dim, output_dim, fusion_type="concat"):
        """Initialize multimodal fusion.
        
        Args:
            vision_dim (int): Dimension of vision features
            audio_dim (int): Dimension of audio features  
            output_dim (int): Target output dimension
            fusion_type (str): Type of fusion ("concat", "add", "gated")
        """
        super(MultimodalFusion, self).__init__()
        
        self.vision_dim = vision_dim
        self.audio_dim = audio_dim
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        
        if fusion_type == "concat":
            # Simple concatenation followed by projection
            self.fusion_proj = torch.nn.Linear(vision_dim + audio_dim, output_dim)
        elif fusion_type == "add":
            # Element-wise addition (requires same dimensions)
            assert vision_dim == audio_dim, f"For add fusion, dimensions must match: {vision_dim} vs {audio_dim}"
            self.vision_proj = torch.nn.Linear(vision_dim, output_dim) if vision_dim != output_dim else torch.nn.Identity()
            self.audio_proj = torch.nn.Linear(audio_dim, output_dim) if audio_dim != output_dim else torch.nn.Identity()
        elif fusion_type == "gated":
            # Gated fusion with learnable weights
            self.vision_proj = torch.nn.Linear(vision_dim, output_dim)
            self.audio_proj = torch.nn.Linear(audio_dim, output_dim)
            self.gate = torch.nn.Linear(output_dim * 2, output_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, vision_features, audio_features):
        """Forward pass for multimodal fusion.
        
        Args:
            vision_features (torch.Tensor): Vision features (B, T, vision_dim)
            audio_features (torch.Tensor): Audio features (B, T, audio_dim)
            
        Returns:
            torch.Tensor: Fused features (B, T, output_dim)
        """
        if self.fusion_type == "concat":
            # Simple concatenation
            fused = torch.cat([vision_features, audio_features], dim=-1)
            fused = self.fusion_proj(fused)
        elif self.fusion_type == "add":
            # Element-wise addition
            vision_proj = self.vision_proj(vision_features)
            audio_proj = self.audio_proj(audio_features)
            fused = vision_proj + audio_proj
        elif self.fusion_type == "gated":
            # Gated fusion
            vision_proj = self.vision_proj(vision_features)
            audio_proj = self.audio_proj(audio_features)
            gate_input = torch.cat([vision_proj, audio_proj], dim=-1)
            gate_weights = torch.sigmoid(self.gate(gate_input))
            fused = gate_weights * vision_proj + (1 - gate_weights) * audio_proj
        
        return fused


class E2E(torch.nn.Module):
    def __init__(self, odim, modality=None, ctc_weight=0.1, ignore_id=-1, 
                 vision_encoder=None, audio_encoder=None, decoder="transformer", 
                 use_qlora=False, qlora_r=16, qlora_alpha=32,
                 vision_model_name=None, audio_model_name=None, decoder_model_name=None,
                 **kwargs):
        super().__init__()

        # Validate input parameters
        if odim is None or odim <= 0:
            raise ValueError(f"Output dimension (odim) must be a positive integer, got {odim}")
        if not (0.0 <= ctc_weight <= 1.0):
            raise ValueError(f"CTC weight must be between 0.0 and 1.0, got {ctc_weight}")

        # Store parameters for validation
        self.use_qlora = use_qlora
        self.qlora_r = qlora_r
        self.qlora_alpha = qlora_alpha
        self.vision_model_name = vision_model_name
        self.audio_model_name = audio_model_name
        self.decoder_model_name = decoder_model_name

        # Validate model combinations
        try:
            self._validate_model_combinations(modality, vision_encoder, audio_encoder, decoder)
        except (ValueError, NotImplementedError) as e:
            logging.error(f"Model configuration validation failed: {e}")
            raise
        
        # Determine modality from encoders if not explicitly provided
        self.modality = self._determine_modality(modality, vision_encoder, audio_encoder)
        
        # Store encoder and decoder types for logging and checkpoints
        self.vision_encoder_type = vision_encoder
        self.audio_encoder_type = audio_encoder
        self.decoder_type = decoder
        self.use_qlora = use_qlora
        self.qlora_r = qlora_r
        self.qlora_alpha = qlora_alpha
        self.vision_model_name = vision_model_name
        self.audio_model_name = audio_model_name
        self.decoder_model_name = decoder_model_name
        
        # Create frontend(s) based on modality and encoder types
        try:
            if self.modality == "audio":
                encoder_type = audio_encoder if audio_encoder else "resnet1d"
                # Handle unfreezing for single modality
                audio_kwargs = kwargs.copy()
                if 'unfreeze_audio' in kwargs:
                    if encoder_type == "whisper":
                        audio_kwargs['freeze_encoder'] = not kwargs['unfreeze_audio']
                    else:
                        audio_kwargs['frozen'] = not kwargs['unfreeze_audio']
                    audio_kwargs.pop('unfreeze_audio', None)
                    audio_kwargs.pop('unfreeze_vision', None)
                
                self.frontend = create_audio_encoder(encoder_type, model_name=audio_model_name, **audio_kwargs)
                # Get output dimension and create projection
                frontend_dim = get_encoder_output_dim(self.frontend, encoder_type, audio_model_name)
                if frontend_dim <= 0:
                    raise ValueError(f"Invalid frontend output dimension: {frontend_dim}")
                self.proj_encoder = torch.nn.Linear(frontend_dim, 768)
                logging.info(f"Audio frontend: {encoder_type} -> {frontend_dim} -> 768")
                
            elif self.modality == "video":
                encoder_type = vision_encoder if vision_encoder else "resnet"
                # Handle unfreezing for single modality
                vision_kwargs = kwargs.copy()
                if 'unfreeze_vision' in kwargs:
                    vision_kwargs['frozen'] = not kwargs['unfreeze_vision']
                    vision_kwargs.pop('unfreeze_vision', None)
                    vision_kwargs.pop('unfreeze_audio', None)
                
                self.frontend = create_vision_encoder(encoder_type, model_name=vision_model_name, **vision_kwargs)
                # Get output dimension and create projection
                frontend_dim = get_encoder_output_dim(self.frontend, encoder_type, vision_model_name)
                if frontend_dim <= 0:
                    raise ValueError(f"Invalid frontend output dimension: {frontend_dim}")
                self.proj_encoder = torch.nn.Linear(frontend_dim, 768)
                logging.info(f"Vision frontend: {encoder_type} -> {frontend_dim} -> 768")
                
            elif self.modality == "multimodal":
                # For multimodal, create both frontends
                vision_type = vision_encoder if vision_encoder else "resnet"
                audio_type = audio_encoder if audio_encoder else "resnet1d"
                
                try:
                    # Pass unfreezing parameter to vision encoder
                    vision_kwargs = kwargs.copy()
                    if 'unfreeze_vision' in kwargs:
                        vision_kwargs['frozen'] = not kwargs['unfreeze_vision']
                        # Remove the unfreeze_vision parameter as encoders don't expect it
                        vision_kwargs.pop('unfreeze_vision', None)
                        vision_kwargs.pop('unfreeze_audio', None)  # Also remove audio param
                    
                    self.vision_frontend = create_vision_encoder(vision_type, model_name=vision_model_name, **vision_kwargs)
                    vision_dim = get_encoder_output_dim(self.vision_frontend, vision_type, vision_model_name)
                    if vision_dim <= 0:
                        raise ValueError(f"Invalid vision frontend output dimension: {vision_dim}")
                except Exception as e:
                    logging.error(f"Failed to create vision frontend: {e}")
                    raise
                
                try:
                    # Pass unfreezing parameter to audio encoder
                    audio_kwargs = kwargs.copy()
                    if 'unfreeze_audio' in kwargs:
                        if audio_type == "whisper":
                            audio_kwargs['freeze_encoder'] = not kwargs['unfreeze_audio']
                        else:
                            audio_kwargs['frozen'] = not kwargs['unfreeze_audio']
                        # Remove the unfreeze parameters as encoders don't expect them
                        audio_kwargs.pop('unfreeze_vision', None)
                        audio_kwargs.pop('unfreeze_audio', None)
                    
                    self.audio_frontend = create_audio_encoder(audio_type, model_name=audio_model_name, **audio_kwargs)
                    audio_dim = get_encoder_output_dim(self.audio_frontend, audio_type, audio_model_name)
                    if audio_dim <= 0:
                        raise ValueError(f"Invalid audio frontend output dimension: {audio_dim}")
                except Exception as e:
                    logging.error(f"Failed to create audio frontend: {e}")
                    raise
                
                # Create multimodal fusion module
                try:
                    self.multimodal_fusion = MultimodalFusion(
                        vision_dim=vision_dim,
                        audio_dim=audio_dim,
                        output_dim=768,
                        fusion_type="concat"  # Simple concatenation for now
                    )
                except Exception as e:
                    logging.error(f"Failed to create multimodal fusion: {e}")
                    raise
                
                logging.info(f"Multimodal fusion: vision {vision_type}({vision_dim}) + audio {audio_type}({audio_dim}) -> 768")
                
            else:
                # Backward compatibility: use original logic
                if modality == "audio":
                    self.frontend = audio_resnet()
                    self.proj_encoder = torch.nn.Linear(512, 768)
                elif modality == "video":
                    self.frontend = video_resnet()
                    self.proj_encoder = torch.nn.Linear(512, 768)
                else:
                    raise ValueError(f"Unknown modality for backward compatibility: {modality}")
        
        except Exception as e:
            logging.error(f"Failed to create frontend components: {e}")
            raise

        # Create Conformer encoder
        try:
            self.encoder = ConformerEncoder(
                attention_dim=768,
                attention_heads=12,
                linear_units=3072,
                num_blocks=12,
                cnn_module_kernel=31,
            )
        except Exception as e:
            logging.error(f"Failed to create Conformer encoder: {e}")
            raise

        # Create decoder using factory function
        try:
            # Filter out encoder-specific parameters that shouldn't be passed to decoder
            decoder_kwargs = kwargs.copy()
            encoder_specific_params = ['unfreeze_vision', 'unfreeze_audio', 'frozen']
            for param in encoder_specific_params:
                decoder_kwargs.pop(param, None)
            
            # For multimodal, pass the fusion output dimension as encoder_dim
            if self.modality == "multimodal":
                decoder_kwargs['encoder_dim'] = 768  # Multimodal fusion output dimension
                self.decoder = create_decoder(decoder, odim=odim, model_name=decoder_model_name, **decoder_kwargs)
            else:
                # For single modality, pass the frontend output dimension
                decoder_kwargs['encoder_dim'] = 768  # Standard conformer dimension
                self.decoder = create_decoder(decoder, odim=odim, model_name=decoder_model_name, **decoder_kwargs)
        except Exception as e:
            logging.error(f"Failed to create decoder: {e}")
            raise

        self.blank = 0
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id

        # Initialize loss functions
        self.ctc_weight = ctc_weight
        try:
            self.ctc = CTC(odim, 768, 0.1, reduce=True)
        except Exception as e:
            logging.error(f"Failed to create CTC loss: {e}")
            raise
        
        try:
            self.criterion = LabelSmoothingLoss(self.odim, self.ignore_id, 0.1, False)
        except Exception as e:
            logging.error(f"Failed to create label smoothing loss: {e}")
            raise
        
        # Apply QLoRA if requested
        if self.use_qlora:
            # Validate QLoRA parameters early
            if self.qlora_r <= 0:
                raise ValueError(f"QLoRA rank (r) must be positive, got {self.qlora_r}")
            if self.qlora_alpha <= 0:
                raise ValueError(f"QLoRA alpha must be positive, got {self.qlora_alpha}")
            
            try:
                self._apply_qlora()
            except Exception as e:
                logging.warning(f"QLoRA application failed: {e}. Continuing with standard training.")
                self.use_qlora = False
        
        # Log the architecture configuration
        try:
            self._log_architecture()
        except Exception as e:
            logging.warning(f"Failed to log architecture configuration: {e}")
            # Continue execution as this is not critical
        
        # Profile model parameters and memory usage (can be disabled with env var)
        if not os.environ.get('DISABLE_MODEL_PROFILING', False):
            try:
                self._profile_model_resources()
            except Exception:
                # Continue execution as this is not critical
                pass
        
        # Validate the final model state
        try:
            self._validate_model_state()
        except Exception as e:
            logging.error(f"Model state validation failed: {e}")
            raise

    def _validate_model_combinations(self, modality, vision_encoder, audio_encoder, decoder):
        """Validate that the specified model combinations are valid."""
        # Valid encoder types
        valid_vision_encoders = ["resnet", "vit", "vivit", "clip-vit"]
        valid_audio_encoders = ["resnet1d", "whisper", "wavlm", "conformer"]
        valid_decoders = ["transformer", "llama", "whisper-decoder"]
        valid_modalities = ["audio", "video", "multimodal"]
        
        # Validate modality
        if modality is not None and modality not in valid_modalities:
            raise ValueError(f"Invalid modality '{modality}'. "
                           f"Valid options: {valid_modalities}")
        
        # Validate vision encoder
        if vision_encoder is not None and vision_encoder not in valid_vision_encoders:
            raise ValueError(f"Invalid vision encoder '{vision_encoder}'. "
                           f"Valid options: {valid_vision_encoders}")
        
        # Validate audio encoder
        if audio_encoder is not None and audio_encoder not in valid_audio_encoders:
            raise ValueError(f"Invalid audio encoder '{audio_encoder}'. "
                           f"Valid options: {valid_audio_encoders}")
        
        # Validate decoder
        if decoder not in valid_decoders:
            raise ValueError(f"Invalid decoder '{decoder}'. "
                           f"Valid options: {valid_decoders}")
        
        # Check for conflicting configurations
        if modality and (vision_encoder or audio_encoder):
            if modality == "audio" and vision_encoder:
                raise ValueError("Cannot specify vision encoder when modality is 'audio'. "
                               "Use audio_encoder instead or set modality to 'video' or 'multimodal'.")
            if modality == "video" and audio_encoder:
                raise ValueError("Cannot specify audio encoder when modality is 'video'. "
                               "Use vision_encoder instead or set modality to 'audio' or 'multimodal'.")
            if modality == "multimodal" and not (vision_encoder and audio_encoder):
                raise ValueError("Multimodal mode requires both vision_encoder and audio_encoder to be specified.")
        
        # Ensure at least one encoder is specified
        if not modality and not vision_encoder and not audio_encoder:
            raise ValueError("Must specify either modality or at least one encoder type. "
                           "Example: --modality video or --vision-encoder resnet")
        
        # Check for potentially problematic combinations
        if decoder in ["llama", "whisper-decoder"] and not self.use_qlora:
            logging.warning(f"Using large decoder '{decoder}' without QLoRA may require significant GPU memory. "
                          f"Consider using --use-qlora flag to reduce memory usage.")
        
        # Check for unimplemented combinations
        unimplemented_encoders = ["conformer"]
        if vision_encoder in unimplemented_encoders:
            raise NotImplementedError(f"Vision encoder '{vision_encoder}' is not yet implemented. "
                                    f"Available options: {[e for e in valid_vision_encoders if e not in unimplemented_encoders]}")
        if audio_encoder in unimplemented_encoders:
            raise NotImplementedError(f"Audio encoder '{audio_encoder}' is not yet implemented. "
                                    f"Available options: {[e for e in valid_audio_encoders if e not in unimplemented_encoders]}")
        
        # Validate model name requirements for certain encoders
        if vision_encoder in ["vit", "clip-vit"] and not hasattr(self, 'vision_model_name'):
            logging.warning(f"Vision encoder '{vision_encoder}' works best with a specific model name. "
                          f"Consider specifying --vision-model-name for better results.")
        
        if audio_encoder in ["whisper", "wavlm"] and not hasattr(self, 'audio_model_name'):
            logging.warning(f"Audio encoder '{audio_encoder}' works best with a specific model name. "
                          f"Consider specifying --audio-model-name for better results.")
        
        if decoder in ["llama", "whisper-decoder"] and not hasattr(self, 'decoder_model_name'):
            logging.warning(f"Decoder '{decoder}' requires a specific model name. "
                          f"Consider specifying --decoder-model-name.")

    def _determine_modality(self, modality, vision_encoder, audio_encoder):
        """Determine the modality based on specified encoders."""
        if modality:
            return modality
        
        if vision_encoder and audio_encoder:
            return "multimodal"
        elif vision_encoder:
            return "video"
        elif audio_encoder:
            return "audio"
        else:
            # Default fallback
            return "video"

    def _apply_qlora(self):
        """Apply QLoRA to the model components."""
        try:
            from espnet.nets.pytorch_backend.qlora_utils import apply_qlora, is_qlora_available, log_qlora_info
        except ImportError as e:
            logging.warning(f"QLoRA utilities not available: {e}. "
                          f"Please ensure qlora_utils.py is implemented. Falling back to standard training.")
            self.use_qlora = False
            return
        
        if not is_qlora_available():
            logging.warning("QLoRA dependencies (peft, bitsandbytes) not available. "
                          "Please install with: pip install peft bitsandbytes. "
                          "Falling back to standard training.")
            self.use_qlora = False
            return
        
        try:
            # Apply QLoRA to decoder (most memory-intensive component)
            if hasattr(self, 'decoder') and self.decoder is not None:
                try:
                    logging.info("Applying QLoRA to decoder")
                    self.decoder = apply_qlora(
                        self.decoder,
                        target_modules=None,  # Use default target modules
                        r=self.qlora_r,
                        alpha=self.qlora_alpha,
                        dropout=0.1,
                        use_4bit=True
                    )
                except Exception as e:
                    logging.warning(f"Failed to apply QLoRA to decoder: {e}")
            
            # Apply QLoRA to vision frontend if it's a large model
            if hasattr(self, 'vision_frontend') and self.vision_frontend is not None:
                if self.vision_encoder_type in ["vit", "clip-vit"]:
                    try:
                        logging.info("Applying QLoRA to vision frontend")
                        self.vision_frontend = apply_qlora(
                            self.vision_frontend,
                            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                            r=self.qlora_r,
                            alpha=self.qlora_alpha,
                            dropout=0.1,
                            use_4bit=True
                        )
                    except Exception as e:
                        logging.warning(f"Failed to apply QLoRA to vision frontend: {e}")
            
            # Apply QLoRA to audio frontend if it's a large model
            if hasattr(self, 'audio_frontend') and self.audio_frontend is not None:
                if self.audio_encoder_type in ["whisper", "wavlm"]:
                    try:
                        logging.info("Applying QLoRA to audio frontend")
                        self.audio_frontend = apply_qlora(
                            self.audio_frontend,
                            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                            r=self.qlora_r,
                            alpha=self.qlora_alpha,
                            dropout=0.1,
                            use_4bit=True
                        )
                    except Exception as e:
                        logging.warning(f"Failed to apply QLoRA to audio frontend: {e}")
            
            # Apply QLoRA to single frontend if present
            if hasattr(self, 'frontend') and self.frontend is not None:
                # Only apply to large models
                if (self.modality == "video" and self.vision_encoder_type in ["vit", "clip-vit"]) or \
                   (self.modality == "audio" and self.audio_encoder_type in ["whisper", "wavlm"]):
                    try:
                        logging.info("Applying QLoRA to frontend")
                        self.frontend = apply_qlora(
                            self.frontend,
                            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                            r=self.qlora_r,
                            alpha=self.qlora_alpha,
                            dropout=0.1,
                            use_4bit=True
                        )
                    except Exception as e:
                        logging.warning(f"Failed to apply QLoRA to frontend: {e}")
            
            # Log QLoRA configuration
            try:
                qlora_config = {
                    'r': self.qlora_r,
                    'alpha': self.qlora_alpha,
                    'dropout': 0.1,
                    'use_4bit': True,
                    'target_modules': ["q_proj", "v_proj", "k_proj", "o_proj"]
                }
                log_qlora_info(self, qlora_config)
            except Exception as e:
                logging.warning(f"Failed to log QLoRA info: {e}")
            
        except Exception as e:
            logging.warning(f"Failed to apply QLoRA: {e}. Falling back to standard training.")
            self.use_qlora = False

    def _validate_model_state(self):
        """Validate that the model is in a consistent state after initialization."""
        # Check that required components exist
        if not hasattr(self, 'encoder') or self.encoder is None:
            raise RuntimeError("Encoder not properly initialized")
        
        if not hasattr(self, 'decoder') or self.decoder is None:
            raise RuntimeError("Decoder not properly initialized")
        
        if not hasattr(self, 'ctc') or self.ctc is None:
            raise RuntimeError("CTC loss not properly initialized")
        
        if not hasattr(self, 'criterion') or self.criterion is None:
            raise RuntimeError("Label smoothing loss not properly initialized")
        
        # Check frontend components based on modality
        if self.modality == "multimodal":
            if not hasattr(self, 'vision_frontend') or self.vision_frontend is None:
                raise RuntimeError("Vision frontend not properly initialized for multimodal mode")
            if not hasattr(self, 'audio_frontend') or self.audio_frontend is None:
                raise RuntimeError("Audio frontend not properly initialized for multimodal mode")
            if not hasattr(self, 'multimodal_fusion') or self.multimodal_fusion is None:
                raise RuntimeError("Multimodal fusion not properly initialized")
        else:
            if not hasattr(self, 'frontend') or self.frontend is None:
                raise RuntimeError(f"Frontend not properly initialized for {self.modality} mode")
            if not hasattr(self, 'proj_encoder') or self.proj_encoder is None:
                raise RuntimeError("Projection encoder not properly initialized")
        
        # Check parameter consistency
        if self.odim <= 0:
            raise RuntimeError(f"Invalid output dimension: {self.odim}")
        
        if not (0.0 <= self.ctc_weight <= 1.0):
            raise RuntimeError(f"Invalid CTC weight: {self.ctc_weight}")
        
        logging.info("Model state validation passed")

    def _log_architecture(self):
        """Log the current architecture configuration."""
        print(f"E2E Model Architecture:")
        print(f"  Modality: {self.modality}")
        if self.vision_encoder_type:
            model_name = f" ({self.vision_model_name})" if self.vision_model_name else ""
            print(f"  Vision Encoder: {self.vision_encoder_type}{model_name}")
        if self.audio_encoder_type:
            model_name = f" ({self.audio_model_name})" if self.audio_model_name else ""
            print(f"  Audio Encoder: {self.audio_encoder_type}{model_name}")
        decoder_name = f" ({self.decoder_model_name})" if self.decoder_model_name else ""
        print(f"  Decoder: {self.decoder_type}{decoder_name}")
        print(f"  CTC Weight: {self.ctc_weight}")
        if self.use_qlora:
            print(f"  QLoRA: Enabled (r={self.qlora_r}, alpha={self.qlora_alpha})")
        else:
            print(f"  QLoRA: Disabled")

    def _profile_model_resources(self):
        """Profile model parameters and memory usage."""
        try:
            # Try to use external profiler first
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
            from model_profiler import print_model_profile
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() and next(self.parameters()).is_cuda else "cpu"
            
            # Print comprehensive profiling information
            print_model_profile(self, device=device, logger=logging.getLogger(__name__))
            
        except ImportError:
            # Fallback to simple profiling
            self._simple_profile_model()
        except Exception:
            # Fallback to simple profiling without logging the error
            self._simple_profile_model()
    
    def _simple_profile_model(self):
        """Simple, clean model profiling."""
        print("\n" + "=" * 60)
        print("MODEL SUMMARY")
        print("=" * 60)
        
        # Overall model stats
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"Total Parameters:     {total_params:>12,}")
        print(f"Trainable Parameters: {trainable_params:>12,} ({trainable_params/total_params:.1%})")
        print(f"Frozen Parameters:    {frozen_params:>12,} ({frozen_params/total_params:.1%})")
        
        # Component breakdown
        print(f"\nComponent Breakdown:")
        print("-" * 45)
        
        components = {}
        if hasattr(self, 'vision_frontend') and self.vision_frontend is not None:
            components['Vision Frontend'] = self.vision_frontend
        if hasattr(self, 'audio_frontend') and self.audio_frontend is not None:
            components['Audio Frontend'] = self.audio_frontend
        if hasattr(self, 'frontend') and self.frontend is not None:
            components['Frontend'] = self.frontend
        if hasattr(self, 'encoder') and self.encoder is not None:
            components['Conformer'] = self.encoder
        if hasattr(self, 'decoder') and self.decoder is not None:
            components['Decoder'] = self.decoder
        
        for name, component in components.items():
            if component is not None:
                comp_total = sum(p.numel() for p in component.parameters())
                comp_trainable = sum(p.numel() for p in component.parameters() if p.requires_grad)
                if comp_trainable == 0:
                    status = "Frozen"
                elif comp_trainable == comp_total:
                    status = "Trainable"
                else:
                    status = f"Mixed ({comp_trainable/1e6:.1f}M trainable)"
                print(f"{name:<18} {comp_total/1e6:>6.1f}M  {status}")
        
        print("=" * 60)

    def get_trainable_parameters(self):
        """Get trainable parameters for optimizer (important for QLoRA).
        
        Returns:
            Iterator[torch.nn.Parameter]: Trainable parameters
        """
        if self.use_qlora:
            from espnet.nets.pytorch_backend.qlora_utils import get_trainable_params
            return get_trainable_params(self)
        else:
            return self.parameters()

    def scorers(self):
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def forward(self, x, lengths, label, audio_x=None, audio_lengths=None):
        if self.modality == "audio":
            lengths = torch.div(lengths, 640, rounding_mode="trunc")

        padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)

        # Handle different modalities
        if self.modality == "multimodal":
            # Process both vision and audio inputs
            vision_features = self.vision_frontend(x)
            
            # Use audio_x if provided, otherwise use x (for backward compatibility)
            audio_input = audio_x if audio_x is not None else x
            audio_features = self.audio_frontend(audio_input)
            
            # Align temporal sequences if needed
            vision_features, audio_features = align_temporal_sequences(
                vision_features, audio_features, method="pad_truncate"
            )
            
            # Apply multimodal fusion
            x = self.multimodal_fusion(vision_features, audio_features)
            
            # Update lengths based on the aligned sequence length
            aligned_length = x.size(1)
            if aligned_length != lengths.max():
                # Adjust lengths proportionally
                length_ratio = aligned_length / lengths.max().float()
                lengths = (lengths.float() * length_ratio).long()
                # Recreate padding mask with new lengths
                padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)
                
        else:
            # Single modality processing - simplified approach
            x = self.frontend(x)
            x = self.proj_encoder(x)
            
            # Update lengths and padding mask based on actual output sequence length
            actual_length = x.size(1)
            lengths = torch.full_like(lengths, actual_length)
            padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)

        x, _ = self.encoder(x, padding_mask)

        # ctc loss
        loss_ctc, ys_hat = self.ctc(x, lengths, label)

        # decoder loss
        ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, _ = self.decoder(ys_in_pad, ys_mask, x, padding_mask)
        loss_att = self.criterion(pred_pad, ys_out_pad)
        loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        acc = th_accuracy(
            pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )

        return loss, loss_ctc, loss_att, acc
