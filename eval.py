import logging
from argparse import ArgumentParser

import torch
import torchaudio
from datamodule.data_module import DataModule
from pytorch_lightning import Trainer


# Set environment variables and logger level
logging.basicConfig(level=logging.WARNING)


def get_trainer(args):
    return Trainer(num_nodes=1, devices=1, accelerator="gpu")


def get_lightning_module(args):
    # Set modules and trainer
    from lightning import ModelModule
    modelmodule = ModelModule(args)
    return modelmodule


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--modality",
        type=str,
        help="Type of input modality (deprecated, use --vision-encoder and/or --audio-encoder instead)",
        choices=["audio", "video"],
    )
    # New encoder arguments
    parser.add_argument(
        "--vision-encoder",
        type=str,
        help="Vision encoder type",
        choices=["resnet", "vit", "vivit", "clip-vit"],
    )
    parser.add_argument(
        "--audio-encoder", 
        type=str,
        help="Audio encoder type",
        choices=["resnet1d", "whisper", "wavlm", "conformer"],
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="transformer",
        help="Decoder type (Default: transformer)",
        choices=["transformer", "llama", "whisper-decoder"],
    )
    parser.add_argument(
        "--use-qlora",
        action="store_true",
        help="Enable QLoRA for memory efficiency during evaluation",
    )
    parser.add_argument(
        "--qlora-r",
        type=int,
        default=16,
        help="LoRA rank (Default: 16)",
    )
    parser.add_argument(
        "--qlora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (Default: 32)",
    )
    parser.add_argument(
        "--vision-model-name",
        type=str,
        help="Specific model name for vision encoder (e.g., google/vit-base-patch16-224)",
    )
    parser.add_argument(
        "--audio-model-name",
        type=str,
        help="Specific model name for audio encoder (e.g., openai/whisper-base)",
    )
    parser.add_argument(
        "--decoder-model-name",
        type=str,
        help="Specific model name for decoder (e.g., meta-llama/Llama-2-7b-hf)",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        help="Root directory of preprocessed dataset",
        required=True,
    )
    parser.add_argument(
        "--test-file",
        default="lrs3_test_transcript_lengths_seg16s.csv",
        type=str,
        help="Filename of testing label list. (Default: lrs3_test_transcript_lengths_seg16s.csv)",
        required=True,
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        help="Path to the pre-trained model",
        required=True,
    )
    parser.add_argument(
        "--decode-snr-target",
        type=float,
        default=999999,
        help="Level of signal-to-noise ratio (SNR)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Flag to use debug level for logging",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="test_results.json",
        help="Path to save the decoded results in JSON format. (Default: test_results.json)",
    )
    args = parser.parse_args()
    
    # Validate and process arguments
    args = validate_and_process_args(args)
    
    return args


def validate_and_process_args(args):
    """Validate argument combinations and implement automatic modality detection."""
    
    # Store original modality for conflict checking
    original_modality = args.modality
    
    # Handle backward compatibility with --modality argument
    if args.modality and not args.vision_encoder and not args.audio_encoder:
        # Legacy mode: use modality to set appropriate encoder
        if args.modality == "video":
            args.vision_encoder = "resnet"
        elif args.modality == "audio":
            args.audio_encoder = "resnet1d"
        print(f"[DEPRECATED] Using --modality={args.modality} is deprecated. "
              f"Consider using --{'vision' if args.modality == 'video' else 'audio'}-encoder instead.")
    
    # Validate that at least one encoder is specified
    if not args.vision_encoder and not args.audio_encoder and not args.modality:
        raise ValueError("Must specify at least one encoder (--vision-encoder or --audio-encoder) "
                        "or use the deprecated --modality argument")
    
    # Check for conflicts between explicitly specified modality and encoder specifications
    if original_modality:
        if original_modality == "audio" and args.vision_encoder:
            raise ValueError("Cannot specify --vision-encoder when --modality=audio")
        if original_modality == "video" and args.audio_encoder:
            raise ValueError("Cannot specify --audio-encoder when --modality=video")
    
    # Automatic modality detection based on specified encoders
    if args.vision_encoder and args.audio_encoder:
        detected_modality = "multimodal"
    elif args.vision_encoder:
        detected_modality = "video"
    elif args.audio_encoder:
        detected_modality = "audio"
    else:
        detected_modality = args.modality  # Fallback to legacy modality
    
    # Set the final modality
    args.modality = detected_modality
    
    # Validate encoder-decoder combinations
    validate_encoder_decoder_combinations(args)
    
    # Log the configuration
    log_model_configuration(args)
    
    return args


def validate_encoder_decoder_combinations(args):
    """Validate that encoder-decoder combinations are supported."""
    
    # Check for missing model names when required
    if args.vision_encoder in ["vit", "vivit", "clip-vit"] and not args.vision_model_name:
        print(f"[WARNING] No --vision-model-name specified for {args.vision_encoder}. "
              f"Will use default model.")
    
    if args.audio_encoder in ["whisper", "wavlm"] and not args.audio_model_name:
        print(f"[WARNING] No --audio-model-name specified for {args.audio_encoder}. "
              f"Will use default model.")
    
    if args.decoder in ["llama", "whisper-decoder"] and not args.decoder_model_name:
        print(f"[WARNING] No --decoder-model-name specified for {args.decoder}. "
              f"Will use default model.")
    
    # Check QLoRA compatibility
    if args.use_qlora:
        qlora_compatible_models = ["llama", "whisper", "wavlm", "vit", "vivit", "clip-vit"]
        incompatible_models = []
        
        if args.vision_encoder and args.vision_encoder not in qlora_compatible_models:
            incompatible_models.append(f"vision encoder '{args.vision_encoder}'")
        if args.audio_encoder and args.audio_encoder not in qlora_compatible_models:
            incompatible_models.append(f"audio encoder '{args.audio_encoder}'")
        if args.decoder not in qlora_compatible_models:
            incompatible_models.append(f"decoder '{args.decoder}'")
        
        if incompatible_models:
            print(f"[WARNING] QLoRA may not be compatible with: {', '.join(incompatible_models)}. "
                  f"Will attempt to apply QLoRA where possible.")


def log_model_configuration(args):
    """Log the final model configuration."""
    print("=" * 50)
    print("EVALUATION MODEL CONFIGURATION")
    print("=" * 50)
    print(f"Modality: {args.modality}")
    
    if args.vision_encoder:
        model_name = f" ({args.vision_model_name})" if args.vision_model_name else ""
        print(f"Vision Encoder: {args.vision_encoder}{model_name}")
    
    if args.audio_encoder:
        model_name = f" ({args.audio_model_name})" if args.audio_model_name else ""
        print(f"Audio Encoder: {args.audio_encoder}{model_name}")
    
    decoder_name = f" ({args.decoder_model_name})" if args.decoder_model_name else ""
    print(f"Decoder: {args.decoder}{decoder_name}")
    
    if args.use_qlora:
        print("QLoRA: Enabled")
    else:
        print("QLoRA: Disabled")
    
    print("=" * 50)


def init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = parse_args()
    init_logger(args.debug)
    modelmodule = get_lightning_module(args)
    datamodule = DataModule(args)
    trainer = get_trainer(args)
    trainer.test(model=modelmodule, datamodule=datamodule)


if __name__ == "__main__":
    cli_main()


## Usage : python eval.py --modality=video --root-dir=/data/ssd2/data_rishabh/lrs2_rf/ --test-file=/data/ssd2/data_rishabh/lrs2_rf/labels/lrs2_test_transcript_lengths_seg16s.csv --pretrained-model-path=/home/rijain@ad.mee.tcd.ie/Experiments/avsr/exp/morell_test1/epoch\=1.ckpt --output-json=./infer/test1.json