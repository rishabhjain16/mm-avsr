import logging
import os
import warnings
from argparse import ArgumentParser

from average_checkpoints import ensemble
from datamodule.data_module import DataModule
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger

# Suppress torchvision video deprecation warnings
warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities.*")
warnings.filterwarnings("ignore", message=".*TorchCodec.*")
# Suppress DDP unused parameters warning (we handle this correctly)
warnings.filterwarnings("ignore", message=".*find_unused_parameters=True.*")


# Set environment variables and logger level
# logging.basicConfig(level=logging.WARNING)
# 

def get_trainer(args):
    seed_everything(42, workers=True)
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.exp_dir, args.exp_name) if args.exp_dir else None,
        monitor="monitoring_step",
        mode="max",
        save_last=True,
        filename="{epoch}",
        save_top_k=10,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]

    return Trainer(
        sync_batchnorm=True,
        default_root_dir=args.exp_dir,
        max_epochs=args.max_epochs,
        num_nodes=args.num_nodes,
        devices=args.gpus,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
        logger=WandbLogger(name=args.exp_name, project="auto_avsr_lipreader", group=args.group_name),
        gradient_clip_val=10.0,
    )


def get_lightning_module(args):
    # Set modules and trainer
    from lightning import ModelModule
    modelmodule = ModelModule(args)
    return modelmodule


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--exp-dir",
        default="./exp",
        type=str,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
        required=True,
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        help="Experiment name",
        required=True,
    )
    parser.add_argument(
        "--group-name",
        type=str,
        help="Group name of the task (wandb API)",
    )
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
        help="Enable QLoRA training for memory efficiency",
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
        "--unfreeze-vision",
        action="store_true",
        help="Unfreeze vision encoder for full training (increases memory usage)",
    )
    parser.add_argument(
        "--unfreeze-audio", 
        action="store_true",
        help="Unfreeze audio encoder for full training (increases memory usage)",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        help="Root directory of preprocessed dataset",
        required=True,
    )
    parser.add_argument(
        "--train-file",
        type=str,
        help="Filename of training label list",
        required=True,
    )
    parser.add_argument(
        "--val-file",
        default="lrs2_val_transcript_lengths_seg16s.csv",
        type=str,
        help="Filename of validation label list. (Default: lrs2_val_transcript_lengths_seg16s.csv)",
    )
    parser.add_argument(
        "--test-file",
        default="lrs2_test_transcript_lengths_seg16s.csv",
        type=str,
        help="Filename of testing label list. (Default: lrs2_test_transcript_lengths_seg16s.csv)",
    )
    parser.add_argument(
        "--num-nodes",
        default=1,
        type=int,
        help="Number of machines used. (Default: 1)",
        required=True,
    )
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of gpus in each machine. (Default: 1)",
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--transfer-frontend",
        action="store_true",
        help="Flag to load the front-end only, works with `pretrained-model`",
    )
    parser.add_argument(
        "--transfer-encoder",
        action="store_true",
        help="Flag to load the weights of encoder, works with `pretrained-model`",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Number of epochs for warmup. (Default: 5)",
    )
    parser.add_argument(
        "--max-epochs",
        default=10,
        type=int,
        help="Number of epochs. (Default: 75)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=2000,
        help="Maximal number of frames in a batch. (Default: 1600)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate. (Default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.03,
        help="Weight decay",
    )
    parser.add_argument(
        "--ctc-weight",
        type=float,
        default=0.1,
        help="CTC weight",
    )
    parser.add_argument(
        "--train-num-buckets",
        type=int,
        default=400,
        help="Bucket size for the training set",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Path of the checkpoint from which training is resumed.",
    )
    parser.add_argument(
        "--slurm-job-id",
        type=float,
        default=0,
        help="Slurm job id",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Flag to use debug level for logging",
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
    
    # Check for unsupported combinations that might cause issues
    unsupported_combinations = []
    
    # Example: Some decoders might not work well with certain encoders
    if args.decoder == "whisper-decoder" and args.vision_encoder in ["vit", "vivit"]:
        unsupported_combinations.append(
            f"Whisper decoder with {args.vision_encoder} vision encoder may have compatibility issues"
        )
    
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
    
    # Warn about unsupported combinations
    for warning in unsupported_combinations:
        print(f"[WARNING] {warning}")
    
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
    print("MODEL CONFIGURATION")
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
        print(f"QLoRA: Enabled (r={args.qlora_r}, alpha={args.qlora_alpha})")
    else:
        print("QLoRA: Disabled")
    
    print("=" * 50)


def init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = parse_args()
    #init_logger(args.debug)
    #args.slurm_job_id = os.environ["SLURM_JOB_ID"]
    modelmodule = get_lightning_module(args)
    datamodule = DataModule(args, train_num_buckets=args.train_num_buckets)
    trainer = get_trainer(args)
    trainer.fit(model=modelmodule, datamodule=datamodule, ckpt_path=args.ckpt_path)
    ensemble(args)


if __name__ == "__main__":
    cli_main()
