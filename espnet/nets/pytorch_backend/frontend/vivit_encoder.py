import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

try:
    from transformers import VivitModel, VivitConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class ViViTEncoder(nn.Module):
    """Video Vision Transformer (ViViT) encoder for video input.
    
    This encoder uses the official ViViT model from transformers library
    specifically designed for video understanding tasks like lip-reading.
    
    Reference: "ViViT: A Video Vision Transformer" (Arnab et al., 2021)
    """
    
    def __init__(
        self,
        model_name: str = "google/vivit-b-16x2-kinetics400",
        frozen: bool = False,
        output_dim: Optional[int] = None,
        num_frames: int = 32,  # ViViT typically uses 32 frames
        temporal_pooling: str = "mean",  # "mean", "max", "cls"
    ):
        """Initialize ViViT encoder.
        
        Args:
            model_name: HuggingFace ViViT model name
            frozen: Whether to freeze the ViViT parameters
            output_dim: Output dimension (if None, uses model's hidden size)
            num_frames: Number of input frames (should match model's expected frames)
            temporal_pooling: How to pool temporal dimension ("mean", "max", "cls")
        """
        super(ViViTEncoder, self).__init__()
        
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers library is required for ViViT encoder. "
                "Install with: pip install transformers"
            )
        
        self.model_name = model_name
        self.frozen = frozen
        self.num_frames = num_frames
        self.temporal_pooling = temporal_pooling
        
        # Load ViViT model
        try:
            self.vivit = VivitModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use float32 for compatibility
                attn_implementation="eager"  # Use eager attention for compatibility
            )
            self.config = self.vivit.config
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ViViT model '{model_name}'. "
                f"Please ensure the model is available or check your internet connection. "
                f"Original error: {e}"
            )
        
        # Get model dimensions
        self.hidden_size = self.config.hidden_size
        self.output_dim = output_dim if output_dim is not None else self.hidden_size
        
        # Freeze parameters if requested
        if self.frozen:
            for param in self.vivit.parameters():
                param.requires_grad = False
        
        # Output projection if needed
        if self.output_dim != self.hidden_size:
            self.projection = nn.Linear(self.hidden_size, self.output_dim)
        else:
            self.projection = None
        
        # Log model info
        logging.info(f"Loaded ViViT model: {model_name}")
        logging.info(f"Expected frames: {self.config.num_frames}")
        logging.info(f"Hidden size: {self.hidden_size}")
        logging.info(f"Output dim: {self.output_dim}")
    
    def forward(self, xs_pad: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            xs_pad: Input tensor of shape (B, T, H, W) or (B, C, T, H, W)
            
        Returns:
            Output tensor of shape (B, T_out, output_dim)
        """
        # Handle input format - ViViT expects (B, T, C, H, W)
        if xs_pad.dim() == 4:
            # (B, T, H, W) -> (B, T, 3, H, W) assuming grayscale, convert to RGB
            xs_pad = xs_pad.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        elif xs_pad.dim() == 5:
            # Check if input is (B, C, T, H, W) and convert to (B, T, C, H, W)
            if xs_pad.size(1) == 3 and xs_pad.size(2) != 3:
                # Input is (B, C, T, H, W), transpose to (B, T, C, H, W)
                xs_pad = xs_pad.transpose(1, 2)
            elif xs_pad.size(2) == 1:
                # (B, T, 1, H, W) -> (B, T, 3, H, W) convert grayscale to RGB
                xs_pad = xs_pad.repeat(1, 1, 3, 1, 1)
            elif xs_pad.size(2) != 3:
                raise ValueError(f"Expected 1 or 3 channels, got {xs_pad.size(2)}")
        else:
            raise ValueError(f"Expected 4D or 5D input, got {xs_pad.dim()}D")
        
        B, T, C, H, W = xs_pad.shape
        
        # Handle temporal dimension - ViViT expects specific number of frames
        expected_frames = self.config.num_frames
        if T != expected_frames:
            # Interpolate or sample frames to match expected number
            if T > expected_frames:
                # Sample frames uniformly
                indices = torch.linspace(0, T - 1, expected_frames).long()
                xs_pad = xs_pad[:, indices, :, :, :]  # Sample along temporal dimension (dim=1)
            else:
                # Interpolate frames using temporal interpolation
                # Reshape to (B*C*H*W, T) for interpolation
                B, T, C, H, W = xs_pad.shape
                xs_reshaped = xs_pad.permute(0, 2, 3, 4, 1).contiguous().view(B * C * H * W, T)
                xs_interpolated = F.interpolate(
                    xs_reshaped.unsqueeze(1),  # (B*C*H*W, 1, T)
                    size=expected_frames,
                    mode='linear',
                    align_corners=False
                ).squeeze(1)  # (B*C*H*W, T_new)
                xs_pad = xs_interpolated.view(B, C, H, W, expected_frames).permute(0, 4, 1, 2, 3)  # (B, T_new, C, H, W)
            
            logging.debug(f"Adjusted frames from {T} to {xs_pad.size(1)} for ViViT")
        
        # Process through ViViT
        with torch.set_grad_enabled(not self.frozen):
            # ViViT expects pixel_values in format (B, T, C, H, W)
            outputs = self.vivit(pixel_values=xs_pad)
            
            # Get the sequence output
            if hasattr(outputs, 'last_hidden_state'):
                # Use sequence output for temporal modeling
                sequence_output = outputs.last_hidden_state  # (B, num_patches, hidden_size)
            elif hasattr(outputs, 'pooler_output'):
                # Use pooled output if sequence not available
                pooled_output = outputs.pooler_output  # (B, hidden_size)
                # Expand to create temporal sequence
                sequence_output = pooled_output.unsqueeze(1)  # (B, 1, hidden_size)
            else:
                raise RuntimeError("ViViT model output format not recognized")
        
        # Apply temporal pooling if we have a sequence
        if sequence_output.size(1) > 1:
            if self.temporal_pooling == "mean":
                # Keep all temporal tokens for sequence modeling
                temporal_output = sequence_output
            elif self.temporal_pooling == "max":
                # Max pooling over temporal dimension
                temporal_output, _ = torch.max(sequence_output, dim=1, keepdim=True)
            elif self.temporal_pooling == "cls":
                # Use first token (CLS token) if available
                temporal_output = sequence_output[:, :1, :]
            else:
                raise ValueError(f"Unknown temporal pooling: {self.temporal_pooling}")
        else:
            temporal_output = sequence_output
        
        # Apply output projection if needed
        if self.projection is not None:
            temporal_output = self.projection(temporal_output)
        
        return temporal_output
    
    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.output_dim


def vivit_encoder(
    model_name: Optional[str] = None,
    frozen: bool = False,
    output_dim: Optional[int] = None,
    num_frames: int = 32,
    temporal_pooling: str = "mean",
    **kwargs
) -> ViViTEncoder:
    """Create ViViT encoder instance.
    
    Args:
        model_name: HuggingFace ViViT model name
        frozen: Whether to freeze ViViT parameters
        output_dim: Output dimension
        num_frames: Number of input frames
        temporal_pooling: How to pool temporal dimension ("mean", "max", "cls")
        **kwargs: Additional arguments
        
    Returns:
        ViViTEncoder instance
    """
    # Set default model name if not provided
    if model_name is None:
        model_name = "google/vivit-b-16x2-kinetics400"
    
    return ViViTEncoder(
        model_name=model_name,
        frozen=frozen,
        output_dim=output_dim,
        num_frames=num_frames,
        temporal_pooling=temporal_pooling,
        **kwargs
    )