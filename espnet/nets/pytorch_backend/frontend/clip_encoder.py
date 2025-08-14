import torch
import torch.nn as nn
from typing import Optional, Union

try:
    from transformers import CLIPVisionModel, CLIPVisionConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class CLIPEncoder(nn.Module):
    """CLIP Vision encoder for video input.
    
    This encoder uses the vision component of CLIP to extract features from video frames.
    Supports both frozen and fine-tunable modes.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        frozen: bool = True,
        output_dim: Optional[int] = None,
        temporal_pooling: str = "mean",
        image_size: int = 224,
        patch_size: int = 32,
    ):
        """Initialize CLIP encoder.
        
        Args:
            model_name: HuggingFace CLIP model name or path
            frozen: Whether to freeze the CLIP parameters (default True for pretrained)
            output_dim: Output dimension (if None, uses model's hidden size)
            temporal_pooling: How to pool temporal dimension ("mean", "max", "last")
            image_size: Input image size
            patch_size: Size of image patches
        """
        super(CLIPEncoder, self).__init__()
        
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers library is required for CLIP encoder. "
                "Install with: pip install transformers"
            )
        
        self.model_name = model_name
        self.frozen = frozen
        self.temporal_pooling = temporal_pooling
        
        # Load CLIP vision model
        try:
            self.clip_vision = CLIPVisionModel.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load CLIP model '{model_name}'. "
                f"Please ensure the model is available or check your internet connection. "
                f"You can download models manually using: "
                f"python -c \"from transformers import CLIPVisionModel; CLIPVisionModel.from_pretrained('{model_name}')\". "
                f"Original error: {e}"
            )
        
        # Get model dimensions
        self.hidden_size = self.clip_vision.config.hidden_size
        self.output_dim = output_dim if output_dim is not None else self.hidden_size
        
        # Freeze parameters if requested
        if self.frozen:
            for param in self.clip_vision.parameters():
                param.requires_grad = False
        
        # Output projection if needed
        if self.output_dim != self.hidden_size:
            self.projection = nn.Linear(self.hidden_size, self.output_dim)
        else:
            self.projection = None
    
    def forward(self, xs_pad: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            xs_pad: Input tensor of shape (B, T, C, H, W) - standard video format
            
        Returns:
            Output tensor of shape (B, T, output_dim)
        """
        # Handle input format - expect (B, T, C, H, W) from dataset
        if xs_pad.dim() == 4:
            # (B, T, H, W) -> (B, T, 1, H, W) - add channel dimension
            xs_pad = xs_pad.unsqueeze(2)
        elif xs_pad.dim() == 5:
            # (B, T, C, H, W) - standard format, no change needed
            pass
        else:
            raise ValueError(f"Expected 4D or 5D input, got {xs_pad.dim()}D")
        
        B, T, C, H, W = xs_pad.shape
        
        # Ensure RGB format for CLIP
        if C == 1:
            # Convert grayscale to RGB by repeating channels
            xs_pad = xs_pad.repeat(1, 1, 3, 1, 1)  # (B, T, 1, H, W) -> (B, T, 3, H, W)
            C = 3
        
        # Reshape to process each frame independently
        # (B, T, C, H, W) -> (B*T, C, H, W)
        xs_reshaped = xs_pad.contiguous().view(B * T, C, H, W)
        
        # Process through CLIP vision encoder
        with torch.set_grad_enabled(not self.frozen):
            outputs = self.clip_vision(pixel_values=xs_reshaped)
            # Use pooled output (equivalent to [CLS] token)
            frame_features = outputs.pooler_output  # (B*T, hidden_size)
        
        # Reshape back to temporal sequence
        # (B*T, hidden_size) -> (B, T, hidden_size)
        frame_features = frame_features.view(B, T, self.hidden_size)
        
        # Apply temporal pooling if needed
        if self.temporal_pooling == "mean":
            # Keep temporal dimension for sequence modeling
            pass
        elif self.temporal_pooling == "max":
            frame_features, _ = torch.max(frame_features, dim=1, keepdim=True)
            frame_features = frame_features.expand(-1, T, -1)
        elif self.temporal_pooling == "last":
            last_frame = frame_features[:, -1:, :]
            frame_features = last_frame.expand(-1, T, -1)
        
        # Apply output projection if needed
        if self.projection is not None:
            frame_features = self.projection(frame_features)
        
        return frame_features
    
    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.output_dim


class CLIPViTEncoder(CLIPEncoder):
    """Alias for CLIPEncoder for consistency with naming conventions."""
    pass


def clip_encoder(
    model_name: str = "openai/clip-vit-base-patch32",
    frozen: bool = True,
    output_dim: Optional[int] = None,
    **kwargs
) -> CLIPEncoder:
    """Create CLIP encoder instance.
    
    Args:
        model_name: HuggingFace CLIP model name
        frozen: Whether to freeze parameters
        output_dim: Output dimension
        **kwargs: Additional arguments
        
    Returns:
        CLIPEncoder instance
    """
    return CLIPEncoder(
        model_name=model_name,
        frozen=frozen,
        output_dim=output_dim,
        **kwargs
    )


def clip_vit_encoder(
    model_name: str = "openai/clip-vit-base-patch32",
    frozen: bool = True,
    output_dim: Optional[int] = None,
    **kwargs
) -> CLIPViTEncoder:
    """Create CLIP ViT encoder instance.
    
    Args:
        model_name: HuggingFace CLIP model name
        frozen: Whether to freeze parameters
        output_dim: Output dimension
        **kwargs: Additional arguments
        
    Returns:
        CLIPViTEncoder instance
    """
    return CLIPViTEncoder(
        model_name=model_name,
        frozen=frozen,
        output_dim=output_dim,
        **kwargs
    )