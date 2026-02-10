"""
Argument Parser for DeepSpeed ZeRO-3 Training with Accelerate
=============================================================

This module provides command-line argument parsing for DeepSpeed
distributed training with hierarchical partitioning support.
"""

import argparse
import os


class AccelerateDeepSpeedArgParser:
    """
    Argument parser for DeepSpeed ZeRO-3 training.

    Includes arguments for:
    - Model and data paths
    - Training hyperparameters
    - DeepSpeed-specific options (hpZ partition size)
    - Profiling and checkpointing
    """

    def __init__(self, description="DeepSpeed ZeRO-3 Training with Accelerate"):
        self.parser = argparse.ArgumentParser(description=description)

        # =====================================================================
        # Model and Data Arguments
        # =====================================================================
        self.parser.add_argument(
            "--model-path",
            type=str,
            required=True,
            help="Path to HuggingFace model directory",
        )
        self.parser.add_argument(
            "--data-path",
            type=str,
            required=True,
            help="Path to training dataset (Alpaca JSON format)",
        )
        self.parser.add_argument(
            "--data-sample",
            type=int,
            default=None,
            help="Subset size for quick testing (default: use full dataset)",
        )

        # =====================================================================
        # Training Hyperparameters
        # =====================================================================
        self.parser.add_argument(
            "--batch-size",
            type=int,
            default=4,
            metavar="N",
            help="Per-device batch size for training (default: 4)",
        )
        self.parser.add_argument(
            "--test-batch-size",
            type=int,
            default=4,
            metavar="N",
            help="Per-device batch size for validation (default: 4)",
        )
        self.parser.add_argument(
            "--epochs",
            type=int,
            default=1,
            metavar="N",
            help="Number of training epochs (default: 1)",
        )
        self.parser.add_argument(
            "--lr",
            type=float,
            default=1e-5,
            metavar="LR",
            help="Learning rate (default: 1e-5)",
        )
        self.parser.add_argument(
            "--lr-warmup",
            type=int,
            default=10,
            metavar="PERCENT",
            help="LR warmup as percentage of total steps (default: 10)",
        )
        self.parser.add_argument(
            "--seed",
            type=int,
            default=1,
            metavar="S",
            help="Random seed for reproducibility (default: 1)",
        )

        # =====================================================================
        # DeepSpeed / Gradient Accumulation
        # =====================================================================
        self.parser.add_argument(
            "--gradient-accumulation-steps",
            type=int,
            default=1,
            help="Number of gradient accumulation steps (default: 1)",
        )
        self.parser.add_argument(
            "--hpz-partition-size",
            type=int,
            default=8,
            help="Number of GPUs per ZeRO partition group (default: 8). "
            "With 32 GPUs and hpz=8: 4 data parallel replicas, each sharded across 8 GPUs.",
        )

        # =====================================================================
        # Validation Settings
        # =====================================================================
        self.parser.add_argument(
            "--no-validation",
            action="store_false",
            dest="run_validation",
            default=True,
            help="Disable validation",
        )
        self.parser.add_argument(
            "--validation-interval",
            type=int,
            default=1,
            help="Run validation every N epochs (default: 1)",
        )

        # =====================================================================
        # Checkpointing
        # =====================================================================
        self.parser.add_argument(
            "--enable-checkpoints",
            action="store_true",
            default=False,
            help="Save checkpoints when validation loss improves",
        )
        self.parser.add_argument(
            "--checkpoints-dir",
            type=str,
            default="/home/bsc/bsc206334/Workspace/"
            + f"distributed-training/accelerate_dist/deepspeed/checkpoints/{os.environ.get('SLURM_JOB_ID', 'local')}",
            help="Directory for saving checkpoints",
        )

        # =====================================================================
        # Profiling and Debugging
        # =====================================================================
        self.parser.add_argument(
            "--profile",
            action="store_true",
            default=False,
            help="Enable PyTorch profiler with NVTX annotations",
        )
        self.parser.add_argument(
            "--profile-logdir",
            type=str,
            default="/home/bsc/bsc206334/Workspace/"
            + f"distributed-training/accelerate_dist/deepspeed/profiler/{os.environ.get('SLURM_JOB_ID', 'local')}",
            help="Directory for profiler output files",
        )
        self.parser.add_argument(
            "--track-memory",
            action="store_true",
            default=False,
            help="Enable CUDA memory history tracking",
        )

        # =====================================================================
        # Experiment Tracking
        # =====================================================================
        self.parser.add_argument(
            "--enable-wandb",
            action="store_true",
            default=False,
            help="Enable Weights & Biases experiment tracking (offline mode)",
        )

        # =====================================================================
        # DataLoader Settings
        # =====================================================================
        self.parser.add_argument(
            "--dataloader-num-workers",
            type=int,
            default=4,
            help="Number of DataLoader workers per process (default: 4)",
        )

        # =====================================================================
        # Model Saving
        # =====================================================================
        self.parser.add_argument(
            "--save-model",
            action="store_true",
            default=False,
            help="Save final model after training",
        )
