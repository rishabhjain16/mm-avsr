#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Simple model management utilities."""

import os
import logging
import subprocess


def get_model_path(model_name):
    """Get the local path for a model.
    
    Args:
        model_name (str): HuggingFace model name (e.g., "meta-llama/Llama-3.2-1B")
        
    Returns:
        str: Path to the model directory
    """
    # Use natural directory structure: meta-llama/Llama-3.2-1B
    model_path = os.path.join(os.getcwd(), "downloaded_models", model_name)
    return model_path


def check_model_exists_locally(model_name):
    """Check if a model exists locally.
    
    Args:
        model_name (str): HuggingFace model name
        
    Returns:
        bool: True if model exists locally with required files
    """
    model_path = get_model_path(model_name)
    
    if not os.path.exists(model_path):
        return False
    
    # Check for essential files
    config_file = os.path.join(model_path, "config.json")
    if not os.path.exists(config_file):
        return False
    
    # Check for at least one model file
    model_files = [
        "pytorch_model.bin",
        "model.safetensors", 
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json"
    ]
    
    has_model_file = any(os.path.exists(os.path.join(model_path, f)) for f in model_files)
    
    # For Whisper models, also check for preprocessor_config.json
    if "whisper" in model_name.lower():
        preprocessor_file = os.path.join(model_path, "preprocessor_config.json")
        if not os.path.exists(preprocessor_file):
            return False
    
    return has_model_file


def download_model_simple(model_name):
    """Simple function to download a model directly using huggingface-cli.
    
    Args:
        model_name (str): HuggingFace model name
        
    Returns:
        str: Path where the model was downloaded
    """
    model_path = get_model_path(model_name)
    
    if check_model_exists_locally(model_name):
        logging.info(f"✅ Model {model_name} already exists at: {model_path}")
        return model_path
    
    logging.info(f"⬇️  Downloading {model_name} to: {model_path}")
    
    try:
        # Try with hf command first (newer)
        result = subprocess.run([
            "hf", "download", model_name, "--local-dir", model_path
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            # Fallback to huggingface-cli
            logging.info("Trying with huggingface-cli...")
            result = subprocess.run([
                "huggingface-cli", "download", model_name, "--local-dir", model_path
            ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logging.info(f"✅ Successfully downloaded {model_name} to: {model_path}")
            return model_path
        else:
            logging.error(f"Download failed: {result.stderr}")
            raise RuntimeError(f"Failed to download {model_name}: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logging.error(f"Download timeout for {model_name}")
        raise RuntimeError(f"Download timeout for {model_name}")
    except FileNotFoundError:
        logging.error("Neither 'hf' nor 'huggingface-cli' command found. Please install huggingface_hub:")
        logging.error("pip install huggingface_hub")
        raise RuntimeError("HuggingFace CLI not found. Install with: pip install huggingface_hub")


def load_model_from_path_or_download(model_class, model_name, **kwargs):
    """Load a model from local path if available, otherwise download it.
    
    Args:
        model_class: The model class to instantiate (e.g., LlamaModel, ViTModel)
        model_name (str): HuggingFace model name
        **kwargs: Additional arguments for model loading
        
    Returns:
        The loaded model instance
    """
    model_path = get_model_path(model_name)
    
    if check_model_exists_locally(model_name):
        logging.info(f"📂 Loading {model_name} from local path: {model_path}")
        return model_class.from_pretrained(model_path, local_files_only=True, **kwargs)
    else:
        logging.info(f"📥 Model {model_name} not found locally. Downloading...")
        try:
            download_model_simple(model_name)
            logging.info(f"📂 Loading {model_name} from downloaded path: {model_path}")
            return model_class.from_pretrained(model_path, local_files_only=True, **kwargs)
        except Exception as e:
            logging.error(f"❌ Failed to download {model_name}: {e}")
            logging.error(f"💡 Please download manually using:")
            logging.error(f"   huggingface-cli download {model_name} --local-dir {model_path}")
            raise


def log_model_info(model_name, model_type="Model"):
    """Log model information and manual download instructions.
    
    Args:
        model_name (str): Name of the model
        model_type (str): Type of model (e.g., "LLaMA", "Whisper", "ViT")
    """
    model_path = get_model_path(model_name)
    
    if check_model_exists_locally(model_name):
        logging.info(f"✅ {model_type} '{model_name}' loaded from: {model_path}")
    else:
        logging.info(f"⬇️  {model_type} '{model_name}' downloaded to: {model_path}")
    
    logging.info(f"💡 To manually download this model:")
    logging.info(f"   huggingface-cli download {model_name} --local-dir {model_path}")
    logging.info(f"   or: hf download {model_name} --local-dir {model_path}")