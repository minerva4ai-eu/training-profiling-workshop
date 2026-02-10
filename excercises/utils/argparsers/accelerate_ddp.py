import argparse
import os
from typing import Optional


class AccelerateDDPArguments:
    def __init__(self, description: str = "PyTorch DDP Example"):
        self.parser = argparse.ArgumentParser(description=description)
        self.parser.add_argument(
            "--model-path",
            type=str,
            help="Path to LLM model",
        )
        self.parser.add_argument(
            "--data-path",
            type=str,
            help="Path to fine-tuning dataset",
        )
        self.parser.add_argument(
            "--batch-size",
            type=int,
            default=4,
            help="Input batch size for training per device (default: 8)",
        )
        self.parser.add_argument(
            "--test-batch-size",
            type=int,
            default=8,
            help="input batch size for testing (default: 1000)",
        )
        self.parser.add_argument(
            "--epochs",
            type=int,
            default=1,
            help="number of epochs to train (default: 3)",
        )
        self.parser.add_argument(
            "--lr",
            type=float,
            default=1e-5,
            metavar="LR",
            help="learning rate (default: .00001)",
        )
        self.parser.add_argument(
            "--lr-warmup",
            type=int,
            default=5,
            help="Learning rate scheduler warmup percentage of steps. Suggested values 5-10%.",
        )
        self.parser.add_argument(
            "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
        )
        self.parser.add_argument(
            "--track-memory",
            action="store_false",
            default=False,
            help="Track the gpu memory",
        )
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
            help="Number of epochs between each validation run",
        )
        self.parser.add_argument(
            "--save-model",
            action="store_false",
            default=True,
            help="For Saving the current Model",
        )
        self.parser.add_argument(
            "--data-sample",
            type=int,
            default=None,
            help="Size of the data sample to use for training",
        )
        self.parser.add_argument(
            "--profile",
            action="store_true",
            default=False,
            help="Enable profiling",
        )
        self.parser.add_argument(
            "--profile-logdir",
            type=str,
            default="/home/bsc/bsc206334/Workspace/"
            + f"distributed-training/accelerate_dist/ddp/profiler/{os.environ['SLURM_JOB_ID']}",
            help="Directory to save profiling logs",
        )
        self.parser.add_argument(
            "--checkpoints-dir",
            type=str,
            default="/home/bsc/bsc206334/Workspace/"
            + f"distributed-training/accelerate_dist/ddp/checkpoints/{os.environ['SLURM_JOB_ID']}",
            help="Directory to save profiling logs",
        )
        self.parser.add_argument(
            "--enable-checkpoints",
            action="store_true",
            default=False,
            help="Enable checkpointing at the end of epochs when validation loss improves",
        )
        self.parser.add_argument(
            "--gradient-accumulation-steps",
            type=int,
            default=1,
            help="Number of steps to accumulate gradients before updating weights (default: 1)",
        )
        self.parser.add_argument(
            "--mixed-precision",
            type=str,
            choices=["no", "fp16", "bf16", "fp8"],
            default="no",
            help="Enable mixed precision training with float16 or bfloat16, if available",
        )
        self.parser.add_argument(
            "--enable-wandb",
            action="store_true",
            default=False,
            help="Enable experiment tracking with Weights & Biases (wandb)",
        )
        self.parser.add_argument(
            "--dataloader-num-workers",
            type=int,
            default=1,
            help="Number of workers for the dataloader (default: 1)",
        )

    def save_json(self, path: Optional[str] = None):
        args = self.parser.parse_args()
        args_dict = vars(args)
        import json

        if path is None:
            path = "args.json"
        with open(path, "w") as f:
            json.dump(args_dict, f, indent=4)
