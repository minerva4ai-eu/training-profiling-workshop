# Libraries used in the distributed training
import datetime
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
import torch.optim as optim
import tqdm
import wandb
from accelerate import Accelerator
from datasets.alpaca.alpaca import AlpacaData
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from transformers.optimization import get_linear_schedule_with_warmup
from utils.argparsers.accelerate_ddp import AccelerateDDPArguments
from utils.exceptions import ProfilingEarlyStop
from utils.utils import (
    accelerate_setup_model,
    cleanup_nccl,
    format_metrics_to_gb,
    gather_across_processes,
    gather_tensor_to_device,
    log_cuda_memory,
    print_rank,
    stack_tensors,
    timeit,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from wandb.wandb_run import Run as WandbRun


@dataclass
class TrainArgs:
    model_name: str
    accelerator: Accelerator
    dataset: AlpacaData
    model: torch.nn.Module
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    optimizer: optim.Optimizer
    scheduler: "LRScheduler"
    epochs: int
    enable_wandb: bool = False
    profile: bool = False
    run_validation: bool = True
    validation_interval: int = 1
    track_memory: bool = False
    enable_checkpoints: bool = False
    checkpoints_dir: str = "./checkpoints"


def validation(
    accelerator, tokenizer, epoch, model, val_loader, max_new_tokens: int = 256
) -> Tuple[dict[str, float], list]:
    model.eval()

    rank = dist.get_rank()
    loss = torch.zeros(3).to(accelerator.device)

    if accelerator.is_main_process:
        inner_pbar = tqdm.tqdm(
            range(len(val_loader)),
            colour="green",
            desc=f"===== Validation Epoch {epoch} =====",
        )

    # =========== Validation loop ===========
    print_rank(rank, f"===== Validation Epoch {epoch} =====")
    local_predictions = []
    local_references = []
    with torch.no_grad():
        # Unwrap model for generation if wrapped by Accelerate
        unwrapped_model = accelerator.unwrap_model(model)

        for batch in val_loader:
            print_rank(rank, f"Validation batch shape: {batch['input_ids'].shape}")

            # Compute validation loss with teacher forcing
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            loss[0] += output["loss"].item()  # sum up batch loss
            loss[1] += +1
            print_rank(
                rank,
                f"Loss: {loss[0] / loss[1]}",
            )

            # Extract input prompt (instruction only) for generation
            # Find where labels start (first non -100 position per sample)
            prompt_lengths = []
            for i, labels in enumerate(batch["labels"]):
                # Find first non-masked position (where labels != -100)
                non_masked = (labels != -100).nonzero(as_tuple=True)[0]
                if len(non_masked) > 0:
                    prompt_lengths.append(non_masked[0].item())
                else:
                    prompt_lengths.append(batch["input_ids"].shape[1])

            print_rank(rank, f"Prompt lengths for generation: {prompt_lengths}")

            # Generate predictions using only the prompt (instruction)
            # Use the shortest prompt length for batched generation
            min_prompt_len = min(prompt_lengths)
            prompt_input_ids = batch["input_ids"][:, :min_prompt_len]
            prompt_attention_mask = batch["attention_mask"][:, :min_prompt_len]

            generated_ids = unwrapped_model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding for reproducibility
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Extract only the generated part (excluding prompt)
            generated_responses = generated_ids[:, min_prompt_len:]

            print_rank(
                rank,
                "Decoding predictions and references for metrics calculation",
            )

            local_predictions.extend(generated_responses)
            local_references.extend(
                stack_tensors(
                    [l[l != -100] for l in batch["labels"]],
                    pad_idx=tokenizer.pad_token_id,
                )
            )

            if accelerator.is_main_process:
                inner_pbar.update(1)

    if accelerator.is_main_process:
        inner_pbar.close()

    print_rank(rank, "Validation loop completed...")
    print_rank(rank, "Gathering loss across processes for final calculation")

    val_loss = (
        accelerator.gather_for_metrics(loss[0]).sum().item()
        / accelerator.gather_for_metrics(loss[1]).sum().item()
    )

    print_rank(0, "Gathering metrics across processes for final calculation")
    local_predictions = stack_tensors(local_predictions, tokenizer.pad_token_id)
    # print_rank(rank, f"Validation | local_predictions: {local_predictions.shape}")
    lpredictions_tokens = tokenizer.batch_decode(
        accelerator.pad_across_processes(
            local_predictions, dim=1, pad_index=tokenizer.pad_token_id
        ),
        skip_special_tokens=True,
    )
    # print_rank(rank, f"local_references: {local_references}")
    local_references = stack_tensors(local_references, tokenizer.pad_token_id)
    # print_rank(rank, f"Validation | local_references.shape: {local_references.shape}")

    # lreferences_tokens = tokenizer.batch_decode(
    #    accelerator.gather(
    #        accelerator.pad_across_processes(
    #            local_references,
    #            dim=1,
    #            pad_index=tokenizer.pad_token_id,
    #        )
    #    ),
    #    skip_special_tokens=True,
    # )
    lreferences_tokens = tokenizer.batch_decode(
        accelerator.pad_across_processes(
            local_references,
            dim=1,
            pad_index=tokenizer.pad_token_id,
        ),
        skip_special_tokens=True,
    )
    for ref, pred in zip(lreferences_tokens, lpredictions_tokens):
        print_rank(rank, f"Ref: {''.join(ref)}")
        print_rank(rank, f"Pred: {''.join(pred)}")

    print_rank(0, f"\tLoss: \t{val_loss:.4f}")

    ref_preds = list(
        zip(
            gather_across_processes(lreferences_tokens),
            gather_across_processes(lpredictions_tokens),
        )
    )

    return {"val_loss": val_loss}, ref_preds


@timeit
def __train_inner_loop(
    accelerator,
    batch,
    model,
    rank,
) -> float:
    with nvtx.range("forward"):
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
    loss = output["loss"]

    print_rank(rank, "Backward...")
    with nvtx.range("backward"):
        accelerator.backward(loss)

    loss_value = loss.detach().item()
    return loss_value


def train_epoch(
    model,
    accelerator: Accelerator,
    epoch: int,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    lr_scheduler: "LRScheduler",
    wandb_run: "WandbRun | None" = None,
    profile=False,
):
    model.train()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()

    num_batches = 0
    epoch_throughput = 0.0
    total_tokens = 0

    if accelerator.is_main_process:
        inner_pbar = tqdm.tqdm(
            range(len(train_loader)),
            colour="blue",
            desc=f"===== Training Epoch {epoch} ===== ",
        )

    # Profiler schedule parameters (from env or defaults)
    skip_first = int(os.environ.get("PROFILE_SKIP_FIRST", "10"))
    wait = int(os.environ.get("PROFILE_WAIT", "1"))
    warmup = int(os.environ.get("PROFILE_WARMUP", "5"))
    active = int(os.environ.get("PROFILE_STEPS_INTERVAL", "10"))

    # Calculate when active window starts and ends
    active_start = skip_first + wait + warmup
    active_end = active_start + active
    profile_early_stop = False  # Whether to stop training after profiling window

    if profile:
        print_rank(0, "Profiler is enabled")
        print_rank(
            0,
            f"Profiler schedule: skip_first={skip_first}, wait={wait}, warmup={warmup}, active={active}",
        )
        print_rank(0, f"Active window: batch {active_start} to {active_end - 1}")

    track_record = {
        "loss": torch.zeros(len(train_loader), dtype=torch.float16, device=local_rank),
        "lr": torch.zeros(len(train_loader), dtype=torch.float16, device=local_rank),
    }

    total_tokens = 0
    num_batches = 0

    nvtx.range_push(f"epoch_{epoch}-rank_{rank}")

    #############################################################################################
    # TODO: Exercise 1: Filling the necessary NVTX ranges and CUDA profiler start/stop calls
    # Example:
    #   1. CUDA API start and end profiler
    #       # make sure to import nvtx at the top of the file:
    #       import torch.cuda.nvtx as nvtx
    #
    #       ...
    #       # Start nsys capture at the beginning of active window
    #       torch.cuda.cudart().cudaProfilerStart()
    #       ...
    #       <code to profile>
    #       with nvtx.range_push("tag of code area"):
    #           <code of tagged code area>
    #       OR
    #       nvtx.range_push("tag of code area")
    #       <code of tagged code area>
    #       nvtx.range_pop()  # for every range_push, there should be a corresponding range_pop
    #       ...
    #
    #       <code to profile>
    #       # Stop nsys capture at the end of active window
    #       torch.cuda.cudart().cudaProfilerStop()
    #       ...
    ##############################################################################################

    train_loader_iter = iter(train_loader)
    for i in range(len(train_loader)):
        if profile and profile_early_stop:
            print_rank(
                rank,
                f"[nsys] Profiler capture complete, exiting training loop at batch {i}",
            )
            dist.barrier()  # Ensure all ranks reach this point before exiting
            raise ProfilingEarlyStop()  # Signal to exit training loop after profiling window

        # Start nsys capture at the beginning of active window
        if profile and i == active_start:
            print_rank(rank, f"[nsys] Starting CUDA profiler capture at batch {i}")

        batch = next(train_loader_iter)

        print_rank(rank, f"Batch {i} (size={len(batch['input_ids'])})")

        ts_start = datetime.datetime.now()
        # Use Accelerate's accumulate() for automatic gradient accumulation
        # This handles: loss scaling, gradient sync skipping, optimizer step timing
        with accelerator.accumulate(model):
            log_cuda_memory()

            print_rank(rank, "Forward...")
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = output["loss"]

            print_rank(rank, "Backward...")
            accelerator.backward(loss)

            loss = loss.detach().item()

            track_record["loss"][i] = loss
            track_record["lr"][i] = optimizer.param_groups[0]["lr"]

            log_cuda_memory()
            print_rank(rank, "Propagating gradients...")
            optimizer.step()

            log_cuda_memory()
            print_rank(rank, "Updating scheduler...")
            lr_scheduler.step()

        log_cuda_memory()

        print_rank(
            rank,
            f"Batch {i} | Loss = {loss} | LR: {lr_scheduler.get_last_lr()[0]}...",
        )

        num_batches += 1
        total_tokens = batch["input_ids"].numel() * dist.get_world_size()
        ts_end = datetime.datetime.now()
        epoch_throughput += total_tokens / (ts_end - ts_start).total_seconds()
        batch_throughput = total_tokens / (ts_end - ts_start).total_seconds()

        # Stop nsys capture at the end of active window
        if profile and i == (active_end - 1):
            print_rank(rank, f"[nsys] Stopping CUDA profiler capture at batch {i}")
            profile_early_stop = True

        if accelerator.is_main_process:
            inner_pbar.update(1)
            current_lr = optimizer.param_groups[0]["lr"]
            inner_pbar.set_postfix(
                loss=f"{loss:.4f}",
                lr=f"{current_lr:.2e}",
                batch_throughput=f"{batch_throughput:.2f} tok/s",
                epoch_throughput=f"{epoch_throughput / num_batches:.2f} tok/s",
            )

    nvtx.range_pop()  # epoch range pop
    if args.enable_wandb:
        # Available only on rank == 0, returns None to other ranks
        all_losses: list[torch.Tensor] = gather_tensor_to_device(
            track_record["loss"],
            gather_device=rank,
        )
        if accelerator.is_main_process:
            for batchid in range(len(train_loader)):
                all_batch_losses = [t[batchid] for t in all_losses]
                batch_avg_loss = sum(all_batch_losses) / len(all_batch_losses)
                wandb_run.log(
                    {
                        "train/loss": batch_avg_loss.item(),
                        "train/lr": track_record["lr"][batchid],
                        "step": batchid,
                    }
                )
            print_rank(0, f"Step {i}: loss = {batch_avg_loss.item():.4f}")

    # =========== End of Training loop ===========

    if accelerator.is_main_process:
        inner_pbar.close()
        print(f"Train Epoch: \t{epoch}, Loss: \t{loss:.4f}")
    print_rank(rank, f"Train Epoch: \t{epoch}, Loss: \t{loss:.4f}")
    return


def train(train_args: TrainArgs):
    best_val_loss = float("inf")
    curr_val_loss = float("inf")

    mem_alloc_tracker = []
    mem_reserved_tracker = []
    ckpt_path = ""

    rank = dist.get_rank()

    if train_args.accelerator.is_main_process and train_args.enable_wandb:
        os.environ["WANDB_MODE"] = "offline"
        wandb_run = wandb.init(
            id=f"accelerate_dist_ddp-nodes-{os.environ.get('NUM_NODES', '1')}-jobid-{os.environ.get('SLURM_JOB_ID', 'local')}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}",
            project=f"Instruct-FT-{train_args.model_name}",
            name=f"accelerate_dist_ddp-nodes-{os.environ.get('NUM_NODES', '1')}-jobid-{os.environ.get('SLURM_JOB_ID', 'local')}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}",
            dir="accelerate_dist/ddp/wandb_logs",
            config={
                "model_name": train_args.model_name,
                "dataset": train_args.dataset.name,
                "epochs": train_args.epochs,
                "batch_size": train_args.train_loader.batch_size,
                "learning_rate": train_args.scheduler.get_last_lr()[0],
            },
        )
    else:
        wandb_run = None

    for epoch in range(train_args.epochs):
        train_epoch(
            accelerator=train_args.accelerator,
            epoch=epoch,
            model=train_args.model,
            train_loader=train_args.train_loader,
            optimizer=train_args.optimizer,
            lr_scheduler=train_args.scheduler,
            profile=train_args.profile,
            wandb_run=wandb_run,
        )

        if (
            train_args.run_validation
            and (epoch + 1) % train_args.validation_interval == 0
        ):
            val_res, ref_preds = validation(
                accelerator=train_args.accelerator,
                tokenizer=train_args.dataset.tokenizer,
                epoch=epoch,
                model=train_args.model,
                val_loader=train_args.val_loader,
            )
            curr_val_loss: float = val_res["val_loss"]

            if train_args.accelerator.is_main_process and train_args.enable_wandb:
                wandb_run.log(
                    {
                        "validation/loss": val_res["val_loss"],
                        "validation/rouge1": val_res["rouge"]["rouge1"],
                        "validation/rouge2": val_res["rouge"]["rouge2"],
                        "validation/rougeL": val_res["rouge"]["rougeL"],
                        "validation/bleu": val_res["bleu"],
                        "validation/epoch": epoch,
                    }
                )

        if train_args.accelerator.is_main_process:
            # for device_refs_preds in ref_preds:
            #    refs, preds = device_refs_preds
            #    for ref, pred in zip(refs, preds):
            #        print_rank(rank, f"Reference: {ref}")
            #        print_rank(rank, f"Prediction: {pred}")

            print_rank(
                rank, f"--> epoch {epoch} completed...entering save and stats zone"
            )

            if train_args.track_memory:
                mem_alloc_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_allocated())
                )
                mem_reserved_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_reserved())
                )
            print_rank(rank, "--> completed save and stats zone...")

        # Save checkpoint if enabled and validation loss improved
        if (
            train_args.enable_checkpoints
            and curr_val_loss < best_val_loss
            and train_args.accelerator.is_main_process
        ):
            print_rank(0, f"-->>>> New Val Loss Record: {curr_val_loss}")

            if os.path.exists(ckpt_path) and ckpt_path != "":
                print_rank(rank, f"Removing previous checkpoint at {ckpt_path}")
                import shutil

                shutil.rmtree(ckpt_path)

            ckpt_path = os.path.join(train_args.checkpoints_dir, f"epoch_{epoch}")
            os.makedirs(train_args.checkpoints_dir, exist_ok=True)
            print_rank(0, "--> entering save model state")

            print_rank(rank, f"-->>>> New Val Loss Record: {curr_val_loss}")
            # save
            ckpt_path = os.path.join(train_args.checkpoints_dir, f"epoch_{epoch}")
            os.makedirs(train_args.checkpoints_dir, exist_ok=True)

            print_rank(rank, "--> entering save model state")

            train_args.accelerator.save_state(output_dir=ckpt_path)
            print_rank(rank, "--> checkpoint state saved")
        best_val_loss = curr_val_loss


def ddp_main(args: "Namespace"):
    train_batch_size = args.batch_size
    model_path = args.model_path
    data_path = args.data_path

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # setup_nccl()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    print_rank(
        rank,
        f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}",
    )
    print_rank(rank, f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    if args.data_sample:
        print_rank(0, f"Using data sample size: {args.data_sample}")
        dataset = AlpacaData(
            data_path,
            model_path,
            sample_size=args.data_sample,
        )
    else:
        dataset = AlpacaData(
            data_path,
            model_path,
        )

    if args.slow_dataloading:
        print_rank(0, "Using slow dataloader settings for profiling...")
        loader_kwargs = {
            "num_workers": 0,
            "pin_memory": False,
        }
        train_loader, val_loader = dataset.get_dataloaders(
            batch_size=args.batch_size,
            loader_args=loader_kwargs,
            pretokenized=False,
        )
    else:
        loader_kwargs = {
            "num_workers": args.dataloader_num_workers,
            "pin_memory": True,
        }
        train_loader, val_loader = dataset.get_dataloaders(
            batch_size=args.batch_size,
            loader_args=loader_kwargs,
        )

    print_rank(rank, "Loading model...")
    # model is on CPU before input to DDP
    model = accelerate_setup_model(accelerator=accelerator, model_path=model_path)

    print_rank(rank, f"Using device: {accelerator.device}")

    print_rank(rank, "Defining optimizer...")
    optimizer = AdamW(params=model.parameters(), lr=args.lr)

    # Calculate actual optimizer steps accounting for gradient accumulation
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_training_steps = steps_per_epoch * args.epochs

    print_rank(
        rank,
        f"Effective batch size: {train_batch_size * world_size * args.gradient_accumulation_steps}",
    )
    print_rank(rank, f"Optimizer steps per epoch: {steps_per_epoch}")

    print_rank(rank, "Defining scheduler...")

    assert args.lr_warmup >= 0 and args.lr_warmup <= 100, (
        "lr_warmup must be between 0 and 100"
    )
    lr_warmup_steps = args.lr_warmup * total_training_steps // 100
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=total_training_steps,
    )

    print_rank(rank, "Running Accelerator prepare...")
    acc_train_loader, acc_val_loader, acc_model, acc_optimizer, acc_scheduler = (
        accelerator.prepare(train_loader, val_loader, model, optimizer, lr_scheduler)
    )

    log_cuda_memory()
    if args.track_memory:
        torch.cuda.memory._record_memory_history(max_entries=100000)

    accelerator.wait_for_everyone()

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
        )
    )

    if args.track_memory:
        try:
            torch.cuda.memory._dump_snapshot(
                os.path.join(
                    os.environ["PROFILE_LOGDIR"],
                    f"cuda_mem_snapshots_device-{dist.get_rank()}.pickle",
                )
            )
        except Exception as e:
            print(f"Exception occured: Failed to capture memory snapshot {e}")
            print("Exception occured: Continueing with training...")

        # Stop recording memory snapshot history.
        torch.cuda.memory._record_memory_history(enabled=None)
    accelerator.wait_for_everyone()
    cleanup_nccl()


if __name__ == "__main__":
    ddp_parser = AccelerateDDPArguments()
    ddp_parser.save_json(os.environ.get("TRAINING_ARGUMENTS_FILE", "ddp_args.json"))

    args = ddp_parser.parser.parse_args()
    if args.slow_dataloading:
        args.dataloader_num_workers = 0

    rank = int(os.environ["RANK"])
    print(f"[ RANK {rank} ]: args.profile : {args.profile}")
    if args.profile and (rank == 0):
        print(f"[ RANK {rank} ]: " + "Profiler is enabled")
        ddp_parser.save_json(os.environ.get("TRAINING_ARGUMENTS_FILE", "ddp_args.json"))

    os.environ["PROFILE_LOGDIR"] = str(args.profile_logdir)

    if args.enable_checkpoints:
        os.environ["CHECKPOINTS_DIR"] = str(args.checkpoints_dir)

    torch.manual_seed(args.seed)

    # print(args)
    try:
        ddp_main(args)
    except ProfilingEarlyStop as e:
        print_rank(
            rank,
            f"Profiling early stop triggered: {e}",
        )
    except Exception as e:
        raise e
