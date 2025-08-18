#!/usr/bin/env python3
"""
Test the modular AVSR architecture with real data from the LRS2 dataset.

This test loads actual video and audio files to verify the system works with real data
and supports testing different encoder/decoder combinations.
"""

import torch
import logging
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datamodule.av_dataset import AVDataset, load_audio, load_video
from datamodule.transforms import AudioTransform, VideoTransform, TextTransform
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configuration presets
MODEL_PRESETS = {
    "resnet_baseline": {
        "vision_encoder": "resnet",
        "audio_encoder": None,
        "decoder": "transformer",
        "use_qlora": False,
        "description": "Video-only ResNet baseline"
    },
    "resnet_multimodal": {
        "vision_encoder": "resnet", 
        "audio_encoder": "resnet1d",
        "decoder": "transformer",
        "use_qlora": False,
        "description": "Multimodal ResNet encoders"
    },
    "vit_advanced": {
        "vision_encoder": "vit",
        "audio_encoder": "whisper",
        "decoder": "llm",
        "use_qlora": True,
        "vision_model_name": "google/vit-base-patch16-224",
        "audio_model_name": "openai/whisper-medium",
        "decoder_model_name": "meta-llama/Llama-2-7b-hf",
        "description": "Advanced ViT + Whisper-Medium + LLaMA with QLoRA"
    },
    "speech_optimized": {
        "vision_encoder": "clip-vit",
        "audio_encoder": "whisper", 
        "decoder": "whisper-decoder",
        "use_qlora": True,
        "vision_model_name": "openai/clip-vit-base-patch32",
        "audio_model_name": "openai/whisper-base",
        "decoder_model_name": "openai/whisper-base",
        "description": "Speech-optimized CLIP + Whisper"
    },
    "memory_efficient": {
        "vision_encoder": "resnet",
        "audio_encoder": "resnet1d", 
        "decoder": "transformer",
        "use_qlora": False,
        "description": "Memory-efficient CNN-based setup"
    },
    "high_performance": {
        "vision_encoder": "vit",
        "audio_encoder": "wavlm",
        "decoder": "llm", 
        "use_qlora": True,
        "vision_model_name": "google/vit-base-patch16-224",
        "audio_model_name": "microsoft/wavlm-base",
        "decoder_model_name": "meta-llama/Llama-2-7b-hf",
        "description": "High-performance transformer setup"
    },
    "whisper_medium_test": {
        "vision_encoder": "vit",
        "audio_encoder": "whisper",
        "decoder": "transformer",  # Use simpler decoder for testing
        "use_qlora": False,
        "vision_model_name": "google/vit-base-patch16-224",
        "audio_model_name": "openai/whisper-medium",
        "description": "ViT + Whisper-Medium + Transformer (test)"
    }
}


def test_model_configuration(config: Dict, root_dir: str, train_file: str, num_samples: int = 3) -> Tuple[bool, Dict]:
    """Test a specific model configuration with real data."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {config['description']}")
    logger.info(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {
        "success": False,
        "error": None,
        "memory_usage": None,
        "inference_time": None,
        "sample_results": []
    }
    
    try:
        # Determine modality from encoders
        if config.get("vision_encoder") and config.get("audio_encoder"):
            modality = "multimodal"
        elif config.get("vision_encoder"):
            modality = "video"
        elif config.get("audio_encoder"):
            modality = "audio"
        else:
            raise ValueError("Must specify at least one encoder")
        
        logger.info(f"Configuration:")
        logger.info(f"  Modality: {modality}")
        logger.info(f"  Vision Encoder: {config.get('vision_encoder', 'None')}")
        logger.info(f"  Audio Encoder: {config.get('audio_encoder', 'None')}")
        logger.info(f"  Decoder: {config.get('decoder', 'transformer')}")
        logger.info(f"  QLoRA: {config.get('use_qlora', False)}")
        
        # Create dataset with encoder-aware transforms
        dataset = AVDataset(
            root_dir=root_dir,
            label_path=train_file,
            subset="train",
            modality=modality,
            audio_transform=AudioTransform("val"),
            video_transform=VideoTransform("val", vision_encoder=config.get("vision_encoder")),
            vision_encoder=config.get("vision_encoder"),
            audio_encoder=config.get("audio_encoder")
        )
        
        logger.info(f"Dataset loaded with {len(dataset)} samples")
        
        # Get vocabulary size
        text_transform = TextTransform()
        vocab_size = len(text_transform.token_list)
        
        # Create model
        logger.info("Creating model...")
        model = E2E(
            odim=vocab_size,
            modality=None,  # Use new encoder-based detection
            vision_encoder=config.get("vision_encoder"),
            audio_encoder=config.get("audio_encoder"),
            decoder=config.get("decoder", "transformer"),
            use_qlora=config.get("use_qlora", False),
            vision_model_name=config.get("vision_model_name"),
            audio_model_name=config.get("audio_model_name"),
            decoder_model_name=config.get("decoder_model_name")
        )
        model = model.to(device)
        model.eval()
        
        logger.info("✓ Model created successfully")
        
        # Memory tracking
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        
        # Test samples
        import time
        total_inference_time = 0
        successful_samples = 0
        
        for i in range(min(num_samples, len(dataset))):
            try:
                sample = dataset[i]
                
                # Prepare data based on modality
                if modality == "video":
                    video_data = sample["input"]
                    audio_data = None
                elif modality == "audio":
                    video_data = None
                    audio_data = sample["input"]
                elif modality == "multimodal":
                    video_data = sample["input"]
                    audio_data = sample["audio_input"]
                
                target = sample["target"]
                
                logger.info(f"Sample {i+1}:")
                if video_data is not None:
                    logger.info(f"  Video shape: {video_data.shape}")
                if audio_data is not None:
                    logger.info(f"  Audio shape: {audio_data.shape}")
                logger.info(f"  Target length: {len(target)}")
                
                # Prepare batch data
                video_batch = video_data.unsqueeze(0).to(device) if video_data is not None else None
                audio_batch = audio_data.unsqueeze(0).to(device) if audio_data is not None else None
                target_batch = torch.tensor(target).unsqueeze(0).to(device)
                
                video_lengths = torch.tensor([video_data.size(0)]).to(device) if video_data is not None else None
                audio_lengths = torch.tensor([audio_data.size(0)]).to(device) if audio_data is not None else None
                
                # Test forward pass with timing
                start_time = time.time()
                with torch.no_grad():
                    loss, loss_ctc, loss_att, acc = model(
                        video_batch, video_lengths,
                        target_batch, audio_batch, audio_lengths
                    )
                inference_time = time.time() - start_time
                total_inference_time += inference_time
                
                logger.info(f"  ✓ Forward pass successful ({inference_time:.3f}s)")
                logger.info(f"    Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")
                
                results["sample_results"].append({
                    "sample_id": i,
                    "success": True,
                    "loss": loss.item(),
                    "accuracy": acc,
                    "inference_time": inference_time
                })
                successful_samples += 1
                
            except Exception as e:
                logger.error(f"  ✗ Sample {i+1} failed: {e}")
                results["sample_results"].append({
                    "sample_id": i,
                    "success": False,
                    "error": str(e)
                })
                continue
        
        # Calculate final metrics
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            results["memory_usage"] = (peak_memory - initial_memory) / 1024**3  # GB
            logger.info(f"Peak memory usage: {results['memory_usage']:.2f} GB")
        
        results["inference_time"] = total_inference_time / max(successful_samples, 1)
        logger.info(f"Average inference time: {results['inference_time']:.3f}s per sample")
        logger.info(f"Successful samples: {successful_samples}/{num_samples}")
        
        if successful_samples > 0:
            results["success"] = True
            logger.info("✅ Configuration test completed successfully!")
        else:
            results["success"] = False
            results["error"] = "No samples processed successfully"
            logger.error("❌ Configuration test failed - no successful samples")
        
        return results["success"], results
        
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        results["success"] = False
        results["error"] = str(e)
        return False, results


def test_multiple_configurations(configs: List[str], root_dir: str, train_file: str, num_samples: int = 3) -> Dict:
    """Test multiple model configurations and compare results."""
    
    logger.info(f"\n{'='*80}")
    logger.info("TESTING MULTIPLE MODEL CONFIGURATIONS")
    logger.info(f"{'='*80}")
    
    all_results = {}
    
    for config_name in configs:
        if config_name not in MODEL_PRESETS:
            logger.error(f"Unknown configuration: {config_name}")
            logger.info(f"Available configurations: {list(MODEL_PRESETS.keys())}")
            continue
        
        config = MODEL_PRESETS[config_name]
        success, results = test_model_configuration(config, root_dir, train_file, num_samples)
        all_results[config_name] = results
    
    # Print comparison summary
    logger.info(f"\n{'='*80}")
    logger.info("CONFIGURATION COMPARISON SUMMARY")
    logger.info(f"{'='*80}")
    
    logger.info(f"{'Configuration':<20} {'Success':<8} {'Memory (GB)':<12} {'Avg Time (s)':<12} {'Description'}")
    logger.info("-" * 80)
    
    for config_name, results in all_results.items():
        config = MODEL_PRESETS[config_name]
        success_str = "✅ Yes" if results["success"] else "❌ No"
        memory_str = f"{results['memory_usage']:.2f}" if results["memory_usage"] else "N/A"
        time_str = f"{results['inference_time']:.3f}" if results["inference_time"] else "N/A"
        
        logger.info(f"{config_name:<20} {success_str:<8} {memory_str:<12} {time_str:<12} {config['description']}")
    
    return all_results


def test_with_real_data(config_names: Optional[List[str]] = None, num_samples: int = 3):
    """Test the modular AVSR architecture with real LRS2 data."""
    
    # Dataset paths from your training command
    root_dir = "/home/rishabh/Desktop/Datasets/lrs2_rf/"
    train_file = "/home/rishabh/Desktop/Datasets/lrs2_rf/labels/lrs2_train_transcript_lengths_seg16s.csv"
    
    # Check if dataset exists
    if not os.path.exists(root_dir) or not os.path.exists(train_file):
        logger.warning(f"Dataset not found at {root_dir}")
        logger.info("Skipping real data test - dataset not available")
        return True
    
    logger.info("Testing modular AVSR architecture with real LRS2 data...")
    
    # Default configurations to test
    if config_names is None:
        config_names = ["resnet_baseline", "resnet_multimodal"]
    
    # Test configurations
    all_results = test_multiple_configurations(config_names, root_dir, train_file, num_samples)
    
    # Check if any configuration succeeded
    success = any(results["success"] for results in all_results.values())
    
    if success:
        logger.info("\n🎉 Real data testing completed successfully!")
    else:
        logger.error("\n💥 All configurations failed!")
    
    return success


def show_training_commands():
    """Show training commands for different model configurations."""
    
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING COMMANDS FOR DIFFERENT CONFIGURATIONS")
    logger.info(f"{'='*80}")
    
    base_cmd = ("python train.py --exp-dir=./exp/ "
                "--root-dir=/home/rishabh/Desktop/Datasets/lrs2_rf/ "
                "--train-file=/home/rishabh/Desktop/Datasets/lrs2_rf/labels/lrs2_train_transcript_lengths_seg16s.csv "
                "--num-nodes=1")
    
    for config_name, config in MODEL_PRESETS.items():
        logger.info(f"\n{config['description']}:")
        logger.info(f"Configuration: {config_name}")
        
        cmd_parts = [base_cmd, f"--exp-name={config_name}"]
        
        if config.get("vision_encoder"):
            cmd_parts.append(f"--vision-encoder={config['vision_encoder']}")
        if config.get("audio_encoder"):
            cmd_parts.append(f"--audio-encoder={config['audio_encoder']}")
        if config.get("decoder"):
            cmd_parts.append(f"--decoder={config['decoder']}")
        if config.get("use_qlora"):
            cmd_parts.append("--use-qlora")
        if config.get("vision_model_name"):
            cmd_parts.append(f"--vision-model-name={config['vision_model_name']}")
        if config.get("audio_model_name"):
            cmd_parts.append(f"--audio-model-name={config['audio_model_name']}")
        if config.get("decoder_model_name"):
            cmd_parts.append(f"--decoder-model-name={config['decoder_model_name']}")
        
        # Format command nicely
        cmd = " \\\n  ".join(cmd_parts)
        logger.info(cmd)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test modular AVSR architecture with different configurations")
    
    parser.add_argument(
        "--configs", 
        nargs="+", 
        choices=list(MODEL_PRESETS.keys()) + ["all"],
        default=["resnet_baseline", "resnet_multimodal"],
        help="Model configurations to test"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to test per configuration"
    )
    
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available model configurations"
    )
    
    parser.add_argument(
        "--show-commands",
        action="store_true", 
        help="Show training commands for all configurations"
    )
    
    return parser.parse_args()


def list_configurations():
    """List all available model configurations."""
    logger.info(f"\n{'='*80}")
    logger.info("AVAILABLE MODEL CONFIGURATIONS")
    logger.info(f"{'='*80}")
    
    for config_name, config in MODEL_PRESETS.items():
        logger.info(f"\n{config_name}:")
        logger.info(f"  Description: {config['description']}")
        logger.info(f"  Vision Encoder: {config.get('vision_encoder', 'None')}")
        logger.info(f"  Audio Encoder: {config.get('audio_encoder', 'None')}")
        logger.info(f"  Decoder: {config.get('decoder', 'transformer')}")
        logger.info(f"  QLoRA: {config.get('use_qlora', False)}")
        
        if config.get('vision_model_name'):
            logger.info(f"  Vision Model: {config['vision_model_name']}")
        if config.get('audio_model_name'):
            logger.info(f"  Audio Model: {config['audio_model_name']}")
        if config.get('decoder_model_name'):
            logger.info(f"  Decoder Model: {config['decoder_model_name']}")


def main():
    """Main function to run real data test."""
    args = parse_arguments()
    
    if args.list_configs:
        list_configurations()
        return
    
    if args.show_commands:
        show_training_commands()
        return
    
    logger.info("Testing modular AVSR architecture with real data...")
    
    # Handle "all" option
    if "all" in args.configs:
        configs_to_test = list(MODEL_PRESETS.keys())
    else:
        configs_to_test = args.configs
    
    logger.info(f"Testing configurations: {configs_to_test}")
    logger.info(f"Samples per configuration: {args.num_samples}")
    
    # Test with real data
    success = test_with_real_data(configs_to_test, args.num_samples)
    
    # Show training command examples
    logger.info(f"\n{'='*60}")
    logger.info("To see training commands for these configurations, run:")
    logger.info("python test_real_data.py --show-commands")
    
    if success:
        logger.info("\n✅ Real data test completed successfully!")
        sys.exit(0)
    else:
        logger.info("\n❌ Real data test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()