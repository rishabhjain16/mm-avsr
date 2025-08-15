import torch
import torch.nn as nn
import logging

try:
    from transformers import WavLMModel, WavLMConfig
    WAVLM_AVAILABLE = True
except ImportError:
    WAVLM_AVAILABLE = False
    logging.warning("transformers not available. WavLMEncoder will not work.")


class WavLMEncoder(nn.Module):
    """WavLM audio encoder frontend.
    
    This class wraps the WavLM encoder to extract audio features
    following the existing audio frontend pattern.
    """
    
    def __init__(
        self,
        model_name="microsoft/wavlm-base",
        freeze_encoder=True,
        layer_extract=-1,
        a_upsample_ratio=1,
    ):
        """Initialize WavLMEncoder.
        
        Args:
            model_name (str): WavLM model name from HuggingFace
            freeze_encoder (bool): Whether to freeze the encoder weights
            layer_extract (int): Which layer to extract features from (-1 for last layer)
            a_upsample_ratio (int): Audio upsampling ratio for temporal resolution
        """
        super(WavLMEncoder, self).__init__()
        
        if not WAVLM_AVAILABLE:
            raise ImportError(
                "transformers is required for WavLMEncoder. "
                "Install with: pip install transformers"
            )
        
        self.model_name = model_name
        self.freeze_encoder = freeze_encoder
        self.layer_extract = layer_extract
        self.a_upsample_ratio = a_upsample_ratio
        
        # Load WavLM model
        from espnet.nets.pytorch_backend.model_cache_utils import (
            load_model_from_path_or_download, log_model_info
        )
        
        self.wavlm = load_model_from_path_or_download(WavLMModel, model_name)
        log_model_info(model_name, "WavLM")
        
        # Get model dimensions
        self.config = self.wavlm.config
        self.hidden_size = self.config.hidden_size
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.wavlm.parameters():
                param.requires_grad = False
            logging.info(f"Frozen WavLM encoder: {model_name}")
        else:
            logging.info(f"Trainable WavLM encoder: {model_name}")
    
    def forward(self, xs_pad):
        """Forward pass.
        
        Args:
            xs_pad (torch.Tensor): Batch of padded input sequences (B, Tmax, idim)
                Expected input is raw audio waveform
        
        Returns:
            torch.Tensor: Encoded features (B, T, hidden_size)
        """
        B, T, C = xs_pad.size()
        
        # WavLM expects raw audio input as (B, T)
        if C == 1:
            # Raw audio input - squeeze the channel dimension
            audio_input = xs_pad.squeeze(-1)  # (B, T)
        else:
            # If multi-channel, take the first channel
            audio_input = xs_pad[:, :, 0]  # (B, T)
        
        # Create attention mask for padded sequences
        # Assume padding is at the end and filled with zeros
        attention_mask = (audio_input != 0).long()
        
        # Process through WavLM
        outputs = self.wavlm(
            input_values=audio_input,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Extract features from specified layer
        if self.layer_extract == -1:
            # Use last layer
            features = outputs.last_hidden_state
        else:
            # Use specific layer
            hidden_states = outputs.hidden_states
            features = hidden_states[self.layer_extract]
        
        return features


class Conv1dWavLMEncoder(nn.Module):
    """WavLM encoder wrapper following the Conv1dResNet pattern."""
    
    def __init__(
        self,
        model_name="microsoft/wavlm-base",
        freeze_encoder=True,
        layer_extract=-1,
        a_upsample_ratio=1,
    ):
        """Initialize Conv1dWavLMEncoder.
        
        Args:
            model_name (str): WavLM model name
            freeze_encoder (bool): Whether to freeze encoder weights
            layer_extract (int): Layer to extract features from
            a_upsample_ratio (int): Audio upsampling ratio
        """
        super(Conv1dWavLMEncoder, self).__init__()
        self.a_upsample_ratio = a_upsample_ratio
        self.trunk = WavLMEncoder(
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


def audio_wavlm_encoder(
    model_name="microsoft/wavlm-base",
    freeze_encoder=True,
    layer_extract=-1,
):
    """Factory function for WavLM audio encoder.
    
    Args:
        model_name (str): WavLM model name
        freeze_encoder (bool): Whether to freeze encoder weights
        layer_extract (int): Layer to extract features from
    
    Returns:
        Conv1dWavLMEncoder: WavLM encoder instance
    """
    # Handle None model_name by using default
    if model_name is None:
        model_name = "microsoft/wavlm-base"
        print(f"[INFO] No WavLM model specified, using default: {model_name}")
    
    return Conv1dWavLMEncoder(
        model_name=model_name,
        freeze_encoder=freeze_encoder,
        layer_extract=layer_extract,
    )