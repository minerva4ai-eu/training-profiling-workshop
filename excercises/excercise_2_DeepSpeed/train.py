import os
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.distributed as dist
import torch.optim as optim
import tqdm
from accelerate import Accelerator
from datasets.alpaca.alpaca import AlpacaData
from torch.cuda import nvtx
from torch.optim.lr_scheduler import LRScheduler
from transformers import (
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from utils.argparsers.accelerate_deepspeed import AccelerateDeepSpeedArgParser
from utils.exceptions import ProfilingEarlyStop
from utils.utils import accelerate2torch_type, cleanup_nccl, log_cuda_memory, print_rank


@dataclass
class TrainArgs:
    """Container for all training arguments passed to training loop."""

    model_name: str
    accelerator: Accelerator
    epochs: int
    dataset: AlpacaData
    model: torch.nn.Module
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    optimizer: torch.optim.Optimizer
    scheduler: LRScheduler
    hpz_partition_size: int
    enable_wandb: bool = False
    profile: bool = False
    run_validation: bool = True
    validation_interval: int = 1
    track_memory: bool = False
    enable_checkpoints: bool = False
    checkpoints_dir: str = None
    activation_checkpointing: bool = False


def accelerate_setup_model(
    accelerator: Accelerator,
    model_path: str,
    tokenizer,
    **kwargs,
):
    """
    Load model with DeepSpeed ZeRO-3 initialization context.

    With ZeRO-3, model parameters are immediately sharded during loading.
    The zero3_init_flag in accelerate config ensures proper initialization.
    """
    rank = int(os.environ["RANK"])
    print_rank(rank, f"Loading model from {model_path}...")

    # Use Accelerate's device placement - DeepSpeed handles sharding
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # attn_implementation="flash_attention_2",
        use_cache=False,  # Disable KV cache for training
        low_cpu_mem_usage=True,
        torch_dtype=accelerate2torch_type[accelerator.mixed_precision],
    )

    # Resize embeddings if tokenizer has more tokens
    model.resize_token_embeddings(len(tokenizer))

    activation_checkpointing = kwargs.get("activation_checkpointing", False)

    # Enable gradient checkpointing for memory efficiency
    if activation_checkpointing:
        model.gradient_checkpointing_enable()
        print_rank(rank, "Gradient checkpointing enabled!")

    print_rank(rank, f"Model loaded with {model.num_parameters():,} parameters")

    return model


def validation(
    train_args: TrainArgs,
    epoch: int,
    print_samples: int = 3,
):
    """
    Run validation loop with DeepSpeed model.
    Gathers predictions from all ranks for complete evaluation.
    """
    rank = int(os.environ["RANK"])

    train_args.model.eval()
    val_loss = 0.0
    num_batches = 0

    tokenizer = train_args.dataset.tokenizer

    # Store local predictions and references
    local_predictions = []
    local_references = []
    local_prompts = []

    print_rank(rank, "Starting validation...")
    train_args.model.eval()

    if train_args.accelerator.is_main_process:
        inner_pbar = tqdm.tqdm(
            range(len(train_args.val_loader)),
            colour="green",
            desc=f"===== Validation Epoch {epoch} =====",
        )

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_args.val_loader):
            nvtx.range_push("val_forward")

            # Compute loss
            print_rank(
                rank, f"Validating batch {batch_idx}/{len(train_args.val_loader)}"
            )
            outputs = train_args.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            nvtx.range_pop()

            loss = outputs.loss
            val_loss += loss.item()
            num_batches += 1

            # Generate predictions for first few batches per rank
            print_rank(rank, f"Generating predictions for batch {batch_idx}")
            if len(local_predictions) < print_samples:
                # Find prompt boundary
                labels = batch["labels"][0]
                prompt_end_idx = (labels != -100).nonzero(as_tuple=True)[0]
                prompt_len = (
                    prompt_end_idx[0].item()
                    if len(prompt_end_idx) > 0
                    else batch["input_ids"].shape[1] // 2
                )

                prompt_ids = batch["input_ids"][0, :prompt_len].unsqueeze(0)
                prompt_mask = batch["attention_mask"][0, :prompt_len].unsqueeze(0)

                # Generate
                print_rank(rank, "Generating text...")
                generated_ids = train_args.model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    max_new_tokens=128,  # Reduced for faster validation
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    synced_gpus=True,  # CRITICAL for DeepSpeed ZeRO-3!
                )

                # Decode
                print_rank(0, "Decoding text...")
                prompt_text = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
                pred_text = tokenizer.decode(
                    generated_ids[0, prompt_len:], skip_special_tokens=True
                )
                ref_text = tokenizer.decode(
                    batch["input_ids"][0, prompt_len:], skip_special_tokens=True
                )

                local_prompts.append(prompt_text)
                local_predictions.append(pred_text)
                local_references.append(ref_text)

                if train_args.accelerator.is_main_process:
                    inner_pbar.update(1)
                    inner_pbar.set_postfix(val_loss=f"{val_loss / num_batches:.4f}")

    if train_args.accelerator.is_main_process:
        inner_pbar.close()
    # ============================================================
    # Gather predictions from all ranks to rank 0
    # ============================================================

    # Use gather_object for variable-length strings
    # all_prompts = [None] * world_size
    # all_predictions = [None] * world_size
    # all_references = [None] * world_size

    # dist.gather_object(local_prompts, all_prompts if rank == 0 else None, dst=0)
    # dist.gather_object(local_predictions, all_predictions if rank == 0 else None, dst=0)
    # dist.gather_object(local_references, all_references if rank == 0 else None, dst=0)

    # Print gathered results on rank 0
    # if rank == 0:
    # Flatten lists: [[rank0_samples], [rank1_samples], ...] -> [all_samples]
    # all_prompts = [item for sublist in all_prompts for item in sublist]
    # all_predictions = [item for sublist in all_predictions for item in sublist]
    # all_references = [item for sublist in all_references for item in sublist]
    all_prompts = local_prompts
    all_predictions = local_predictions
    all_references = local_references
    for i, (prompt, pred, ref) in enumerate(
        zip(
            all_prompts[:print_samples],
            all_predictions[:print_samples],
            all_references[:print_samples],
        )
    ):
        print_rank(0, f"\n{'=' * 60}")
        print_rank(0, f"Sample {i + 1} of {print_samples}")
        print_rank(0, f"{'=' * 60}")
        print_rank(0, f"PROMPT:\n{prompt[:300]}...")
        print_rank(0, f"\nGROUND TRUTH:\n{ref[:300]}")
        print_rank(0, f"\nMODEL OUTPUT:\n{pred[:300]}")

    # Gather losses (you already do this correctly)
    val_loss_tensor = torch.tensor([val_loss], device=train_args.accelerator.device)
    num_batches_tensor = torch.tensor(
        [num_batches], device=train_args.accelerator.device
    )

    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)

    avg_val_loss = val_loss_tensor.item() / num_batches_tensor.item()
    print_rank(0, f"Validation Loss: {avg_val_loss:.4f}")

    train_args.model.train()
    return avg_val_loss


def train_epoch(train_args: TrainArgs, epoch: int):
    """
    Train for one epoch with DeepSpeed.

    DeepSpeed handles:
    - Parameter gathering during forward
    - Gradient reduction during backward
    - Optimizer state management
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    train_args.model.train()
    total_loss = 0.0
    num_batches = 0
    epoch_throughput = 0.0
    total_tokens = 0

    # Wrap dataloader with tqdm progress bar for rank 0 only
    if train_args.accelerator.is_main_process:
        inner_pbar = tqdm.tqdm(
            range(len(train_args.train_loader)),
            colour="blue",
            desc=f"===== Training Epoch {epoch} =====",
            dynamic_ncols=True,
        )

    # Profiler schedule parameters (from env or defaults)
    skip_first = int(os.environ.get("PROFILE_SKIP_FIRST", "10"))
    wait = int(os.environ.get("PROFILE_WAIT", "1"))
    warmup = int(os.environ.get("PROFILE_WARMUP", "5"))
    active = int(os.environ.get("PROFILE_STEPS_INTERVAL", "10"))

    # Calculate when active window starts and ends
    active_start = skip_first + wait + warmup
    active_end = active_start + active

    profile_early_stop = False

    if train_args.profile:
        print_rank(0, "Profiler is enabled")
        print_rank(
            0,
            f"Profiler schedule: skip_first={skip_first}, wait={wait}, warmup={warmup}, active={active}",
        )
        print_rank(0, f"Active window: batch {active_start} to {active_end - 1}")

    train_loader_iter = iter(train_args.train_loader)
    for batch_idx in range(len(train_args.train_loader)):
        if train_args.profile and profile_early_stop:
            print_rank(
                rank,
                f"[nsys] Profiler capture complete, exiting training loop at batch {batch_idx}",
            )
            dist.barrier()  # Ensure all ranks reach this point before exiting
            raise ProfilingEarlyStop()  # Signal to exit training loop after profiling window

        # Start nsys capture at the beginning of active window
        if train_args.profile and batch_idx == active_start:
            print_rank(
                rank, f"[nsys] Starting CUDA profiler capture at batch {batch_idx}"
            )
            torch.cuda.cudart().cudaProfilerStart()

        nvtx.range_push(f"DataLoad batch {batch_idx}")
        batch = next(train_loader_iter)
        nvtx.range_pop()

        ts_start = datetime.now()
        nvtx.range_push(f"batch_{batch_idx}")

        with train_args.accelerator.accumulate(train_args.model):
            log_cuda_memory("Before forward pass |")
            # Forward pass
            nvtx.range_push("forward")
            outputs = train_args.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            nvtx.range_pop()  # forward

            # Backward pass - DeepSpeed handles gradient accumulation and reduction
            log_cuda_memory("Before backward pass |")
            nvtx.range_push("backward")
            train_args.accelerator.backward(loss)
            nvtx.range_pop()  # backward

            # Optimizer step - DeepSpeed handles internally
            log_cuda_memory("Before optimizer step |")
            nvtx.range_push("optimizer_step")
            train_args.optimizer.step()
            train_args.scheduler.step()
            log_cuda_memory("After optimizer step |")

        nvtx.range_pop()  # optimizer_step

        nvtx.range_pop()  # batch_N

        ts_end = datetime.now()

        total_tokens = batch["input_ids"].numel()
        total_loss += loss.item()
        num_batches += 1
        total_tokens = batch["input_ids"].numel()
        epoch_throughput += total_tokens / (ts_end - ts_start).total_seconds()
        batch_throughput = total_tokens / (ts_end - ts_start).total_seconds()

        # Stop nsys capture at the end of active window
        if train_args.profile and batch_idx == (active_end - 1):
            print_rank(
                rank, f"[nsys] Stopping CUDA profiler capture at batch {batch_idx}"
            )
            torch.cuda.cudart().cudaProfilerStop()
            profile_early_stop = True

        # Update tqdm progress bar with current metrics (rank 0 only)
        if train_args.accelerator.is_main_process:
            inner_pbar.update(1)
            # dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)
            current_lr = train_args.optimizer.param_groups[0]["lr"]
            inner_pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{current_lr:.4e}",
                throughput_rank0=f"{batch_throughput:.2f} tok/s",
                avg_throughput_rank0=f"{epoch_throughput / num_batches:.2f} tok/s",
                overall_throughput=f"{batch_throughput * world_size:.2f} tok/s (world size {world_size})",
            )
    if train_args.accelerator.is_main_process:
        inner_pbar.close()

    avg_epoch_throughput = epoch_throughput / num_batches

    avg_train_loss = total_loss / num_batches
    print_rank(0, f"Epoch {epoch} - Average Training Loss: {avg_train_loss:.4f}")
    print_rank(
        0,
        f"Epoch {epoch} - Average Throughput: {avg_epoch_throughput * world_size:.2f} tokens/sec (world size {world_size})",
    )

    return avg_train_loss


def train(train_args: TrainArgs):
    """
    Main training loop with validation and checkpointing.
    """
    rank = int(os.environ["RANK"])
    best_val_loss = float("inf")

    # Initialize WandB if enabled
    if train_args.enable_wandb and rank == 0:
        import wandb

        wandb.init(
            project="deepspeed-training",
            name=f"{train_args.model_name}-deepspeed-zero3",
            config={
                "model": train_args.model_name,
                "epochs": train_args.epochs,
                "world_size": int(os.environ["WORLD_SIZE"]),
            },
            mode="offline",
            dir="accelerate_dist/deepspeed/wandb_logs",
        )

    for epoch in range(1, train_args.epochs + 1):
        nvtx.range_push(f"epoch_{epoch}-rank_{rank}")

        print_rank(0, f"\n{'=' * 60}")
        print_rank(0, f"Epoch {epoch}/{train_args.epochs}")
        print_rank(0, f"{'=' * 60}")

        # Training
        if epoch > 1:
            train_args.profile = False  # Disable profiling after first epoch
        train_loss = train_epoch(train_args, epoch)

        # Validation
        if (
            train_args.run_validation
            and (epoch % train_args.validation_interval == 0)
            and epoch > 0
        ):
            curr_val_loss = validation(train_args, epoch)

            # Log to WandB
            if train_args.enable_wandb and rank == 0:
                import wandb

                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": curr_val_loss,
                        "learning_rate": train_args.optimizer.param_groups[0]["lr"],
                    }
                )

            # Checkpointing
            if train_args.enable_checkpoints and curr_val_loss < best_val_loss:
                print_rank(
                    rank,
                    f"Validation improved: {best_val_loss:.4f} → {curr_val_loss:.4f}",
                )

                ckpt_path = os.path.join(
                    train_args.checkpoints_dir,
                    f"epoch_{epoch}_valloss_{curr_val_loss:.4f}",
                )

                # DeepSpeed checkpointing through Accelerate
                train_args.accelerator.save_state(output_dir=ckpt_path)
                print_rank(rank, f"Checkpoint saved to {ckpt_path}")

                best_val_loss = curr_val_loss

        nvtx.range_pop()  # epoch_N

        train_args.accelerator.wait_for_everyone()

    # Finish WandB
    if train_args.enable_wandb and rank == 0:
        import wandb

        wandb.finish()


def deepspeed_main(args: Namespace):
    """
    Main entry point for DeepSpeed training.
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    hpz_partition_size = int(os.getenv("HPZ_PARTITION_SIZE", 1))
    if hpz_partition_size <= 1:
        raise ValueError(
            (
                "HPZ_PARTITION_SIZE must be greater than 1 for ZeRO-3 with hpZ."
                + "Set it to the number of GPUs that should share sharded parameters."
            )
        )

    # Initialize Accelerator with DeepSpeed
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    print_rank(rank, "Initializing DeepSpeed training...")
    print_rank(rank, f"Rank {rank} / World Size {world_size}")
    print_rank(
        rank,
        f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}",
    )

    print_rank(0, f"ACCELERATE_CONFIG_FILE = {os.environ['ACCELERATE_CONFIG_FILE']}")
    print_rank(0, "DeepSpeed ZeRO Stage: 3 with hpZ")

    # Load dataset
    print_rank(rank, "Loading dataset...")
    if args.data_sample:
        print_rank(0, f"Using data sample size: {args.data_sample}")
        dataset = AlpacaData(
            args.data_path,
            args.model_path,
            sample_size=args.data_sample,
        )
    else:
        dataset = AlpacaData(
            args.data_path,
            args.model_path,
        )

    # Create distributed data loaders
    loader_kwargs = {
        "num_workers": args.dataloader_num_workers,
        "pin_memory": True,
        "shuffle": True,
    }
    train_loader, val_loader = dataset.get_dataloaders(
        batch_size=args.batch_size, loader_args=loader_kwargs, pretokenized=True
    )
    print_rank(rank, "Distributed data loaders created.")
    print_rank(
        rank,
        f"Train batches: {len(train_loader)} | Rank total samples: {len(train_loader.dataset)}",
    )
    print_rank(
        rank,
        f"Val batches: {len(val_loader)} | Rank total samples: {len(val_loader.dataset)}",
    )

    # Load model
    print_rank(rank, "Setting up model...")
    model = accelerate_setup_model(
        accelerator=accelerator,
        model_path=args.model_path,
        tokenizer=dataset.tokenizer,
        activation_checkpointing=args.activation_checkpointing,
    )

    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Calculate training steps
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_training_steps = steps_per_epoch * args.epochs

    print_rank(
        rank,
        f"Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}",
    )
    print_rank(rank, f"Optimizer steps per epoch: {steps_per_epoch}")

    # Setup learning rate scheduler
    assert args.lr_warmup >= 0 and args.lr_warmup <= 100, (
        "lr_warmup must be between 0 and 100"
    )
    lr_warmup_steps = args.lr_warmup * total_training_steps // 100
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=total_training_steps,
    )

    # Prepare with Accelerate (DeepSpeed wrapping happens here)
    print_rank(rank, "Running Accelerator prepare (DeepSpeed initialization)...")
    acc_train_loader, acc_val_loader, acc_model, acc_optimizer, acc_scheduler = (
        accelerator.prepare(train_loader, val_loader, model, optimizer, lr_scheduler)
    )
    print_rank(rank, "Dataloaders after Accelerator prepare:")
    print_rank(
        rank,
        f"Train batches: {len(acc_train_loader)} | Rank total samples: {len(acc_train_loader.dataset)}",
    )
    print_rank(
        rank,
        f"Val batches: {len(acc_val_loader)} | Rank total samples: {len(acc_val_loader.dataset)}",
    )

    log_cuda_memory()

    if args.track_memory:
        torch.cuda.memory._record_memory_history(max_entries=100000)

    accelerator.wait_for_everyone()

    # Start training
    print_rank(rank, "Beginning training...")
    train(
        TrainArgs(
            model_name=args.model_path.split("/")[-1],
            accelerator=accelerator,
            epochs=args.epochs,
            dataset=dataset,
            model=acc_model,
            train_loader=acc_train_loader,
            val_loader=acc_val_loader,
            optimizer=acc_optimizer,
            scheduler=acc_scheduler,
            enable_wandb=args.enable_wandb,
            profile=args.profile,
            run_validation=args.run_validation,
            validation_interval=args.validation_interval,
            track_memory=args.track_memory,
            enable_checkpoints=args.enable_checkpoints,
            checkpoints_dir=args.checkpoints_dir,
            hpz_partition_size=hpz_partition_size,
        )
    )

    # Memory snapshot export
    if args.track_memory:
        try:
            snapshot_path = os.path.join(
                os.environ["PROFILE_LOGDIR"],
                f"cuda_mem_snapshots_device-{dist.get_rank()}.pickle",
            )
            torch.cuda.memory._dump_snapshot(snapshot_path)
        except Exception as e:
            print(f"Failed to capture memory snapshot: {e}")
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)

    accelerator.wait_for_everyone()
    cleanup_nccl()


if __name__ == "__main__":
    deepspeed_parser = AccelerateDeepSpeedArgParser()
    deepspeed_parser.save_json(
        os.environ.get("TRAINING_ARGUMENTS_FILE", "deepspeed_train_args.json")
    )
    args = deepspeed_parser.parser.parse_args()

    rank = os.environ["RANK"]

    # Setup profiling directory
    if args.profile and (int(os.environ["RANK"]) == 0):
        print(f"[ RANK {rank} ]: Profiler is enabled")
        deepspeed_parser.save_json(
            os.environ.get("TRAINING_ARGUMENTS_FILE", "deepspeed_train_args.json")
        )

    if args.enable_checkpoints:
        os.environ["CHECKPOINTS_DIR"] = str(args.checkpoints_dir)

    torch.manual_seed(args.seed)

    try:
        deepspeed_main(args)
    except ProfilingEarlyStop as e:
        print_rank(
            rank,
            f"Profiling early stop triggered: {e}",
        )
    except Exception as e:
        raise e
