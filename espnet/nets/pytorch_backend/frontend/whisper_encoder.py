import torch
import torch.nn as nn
import logging

try:
    from transformers import WhisperModel, WhisperConfig
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logging.warning("transformers not available. WhisperEncoder will not work.")


class WhisperEncoder(nn.Module):
    """Whisper audio encoder frontend.
    
    This class wraps the Whisper encoder to extract audio features
    similar to the existing ResNet1D frontend pattern.
    """
    
    def __init__(
        self,
        model_name="openai/whisper-base",
        freeze_encoder=True,
        layer_extract=-1,
        a_upsample_ratio=1,
    ):
        """Initialize WhisperEncoder.
        
        Args:
            model_name (str): Whisper model name from HuggingFace
            freeze_encoder (bool): Whether to freeze the encoder weights
            layer_extract (int): Which layer to extract features from (-1 for last layer)
            a_upsample_ratio (int): Audio upsampling ratio for temporal resolution
        """
        super(WhisperEncoder, self).__init__()
        
        if not WHISPER_AVAILABLE:
            raise ImportError(
                "transformers is required for WhisperEncoder. "
                "Install with: pip install transformers"
            )
        
        self.model_name = model_name
        self.freeze_encoder = freeze_encoder
        self.layer_extract = layer_extract
        self.a_upsample_ratio = a_upsample_ratio
        
        # Load Whisper model
        from espnet.nets.pytorch_backend.model_cache_utils import (
            load_model_from_path_or_download, log_model_info
        )
        
        self.whisper = load_model_from_path_or_download(WhisperModel, model_name)
        self.encoder = self.whisper.encoder
        log_model_info(model_name, "Whisper encoder")
        
        # Get model dimensions
        self.config = self.whisper.config
        self.hidden_size = self.config.d_model
        
        # Freeze the entire model first (including decoder), then unfreeze encoder if needed
        # This prevents the decoder from consuming memory and gradients
        for param in self.whisper.parameters():
            param.requires_grad = False
        
        # Then handle encoder freezing
        if not freeze_encoder:
            # Only unfreeze encoder if explicitly requested
            for param in self.encoder.parameters():
                param.requires_grad = True
            logging.info(f"Trainable Whisper encoder: {model_name}")
        else:
            logging.info(f"Frozen Whisper encoder: {model_name}")
        
        # Log parameter counts for verification
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.whisper.parameters() if p.requires_grad)
        logging.info(f"Whisper encoder trainable parameters: {encoder_params:,} / {total_params:,}")
    
    def forward(self, xs_pad):
        """Forward pass.
        
        Args:
            xs_pad (torch.Tensor): Batch of padded input sequences (B, Tmax, idim)
                Expected input is raw audio waveform
        
        Returns:
            torch.Tensor: Encoded features (B, T, hidden_size)
        """
        B, T, C = xs_pad.size()
        
        # Whisper expects audio input to be (B, T) for raw waveform
        # or (B, n_mels, T) for mel spectrogram
        if C == 1:
            # Raw audio input - squeeze the channel dimension
            audio_input = xs_pad.squeeze(-1)  # (B, T)
        else:
            # Assume mel spectrogram input
            audio_input = xs_pad.transpose(1, 2)  # (B, C, T)
        
        # Process through Whisper encoder
        if C == 1:
            # For raw audio, we need to convert to mel spectrogram first
            # Use Whisper's feature extractor to convert raw audio to mel-spectrogram
            try:
                from transformers import WhisperFeatureExtractor
                # Load feature extractor from the same location as the model
                from espnet.nets.pytorch_backend.model_cache_utils import get_model_path, check_model_exists_locally
                model_path = get_model_path(self.model_name)
                
                if check_model_exists_locally(self.model_name):
                    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path, local_files_only=True)
                else:
                    feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_name)
                
                # Convert raw audio to mel-spectrogram
                # audio_input shape: (B, T) - raw audio
                mel_features = []
                for i in range(B):
                    # Extract single audio sample and convert to numpy
                    audio_sample = audio_input[i].cpu().numpy()
                    # Feature extractor expects sampling rate 16000
                    mel_spec = feature_extractor(
                        audio_sample, 
                        sampling_rate=16000, 
                        return_tensors="pt"
                    )["input_features"]  # Shape: (1, n_mels, T_mel)
                    mel_features.append(mel_spec.squeeze(0))  # Remove batch dim: (n_mels, T_mel)
                
                # Stack mel features and move to device
                mel_features = torch.stack(mel_features).to(xs_pad.device)  # (B, n_mels, T_mel)
                audio_input = mel_features
                
            except ImportError:
                raise ImportError("WhisperFeatureExtractor required for raw audio input. Install with: pip install transformers")
            
            encoder_outputs = self.encoder(
                input_features=audio_input,
                attention_mask=None,
                head_mask=None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            # For mel spectrogram input
            encoder_outputs = self.encoder(
                input_features=audio_input,
                attention_mask=None,
                head_mask=None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # Extract features from specified layer
        if self.layer_extract == -1:
            # Use last layer
            features = encoder_outputs.last_hidden_state
        else:
            # Use specific layer
            hidden_states = encoder_outputs.hidden_states
            features = hidden_states[self.layer_extract]
        
        return features


class Conv1dWhisperEncoder(nn.Module):
    """Whisper encoder wrapper following the Conv1dResNet pattern."""
    
    def __init__(
        self,
        model_name="openai/whisper-base",
        freeze_encoder=True,
        layer_extract=-1,
        a_upsample_ratio=1,
    ):
        """Initialize Conv1dWhisperEncoder.
        
        Args:
            model_name (str): Whisper model name
            freeze_encoder (bool): Whether to freeze encoder weights
            layer_extract (int): Layer to extract features from
            a_upsample_ratio (int): Audio upsampling ratio
        """
        super(Conv1dWhisperEncoder, self).__init__()
        self.a_upsample_ratio = a_upsample_ratio
        self.trunk = WhisperEncoder(
            model_name=model_name,
            freeze_encoder=freeze_encoder,
            layer_extract=layer_extract,
            a_upsample_ratio=a_upsample_ratio,
        )
    
    def forward(self, xs_pad):
        """Forward pass.
        
        Args:
            xs_pad (torch.Tensor): Batch of padded input sequences (B, Tmax, idim)
        
        Returns:
            torch.Tensor: Encoded features (B, T, hidden_size)
        """
        return self.trunk(xs_pad)


def audio_whisper_encoder(
    model_name="openai/whisper-base",
    freeze_encoder=True,
    layer_extract=-1,
):
    """Factory function for Whisper audio encoder.
    
    Args:
        model_name (str): Whisper model name
        freeze_encoder (bool): Whether to freeze encoder weights
        layer_extract (int): Layer to extract features from
    
    Returns:
        Conv1dWhisperEncoder: Whisper encoder instance
    """
    # Handle None model_name by using default
    if model_name is None:
        model_name = "openai/whisper-base"
        print(f"[INFO] No Whisper model specified, using default: {model_name}")
    
    return Conv1dWhisperEncoder(
        model_name=model_name,
        freeze_encoder=freeze_encoder,
        layer_extract=layer_extract,
    )