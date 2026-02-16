import functools
import os
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.cuda as cu
import torch.distributed as dist
from transformers import AutoModelForCausalLM

if TYPE_CHECKING:
    from accelerate import Accelerator
    from transformers import PreTrainedTokenizer


accelerate2torch_type = {
    "no": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8": torch.float8_e8m0fnu,
}


def timeit(func):
    """Decorator that times the execution of a function and prints the elapsed time."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start timing
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time  # Compute elapsed time
        print_rank(
            os.environ["RANK"],
            f"⏱️ Function '{func.__name__}' executed in {elapsed_time:.6f} seconds",
        )
        return result

    return wrapper


def print_rank(rank, msg):
    """Prints the message with the rank number"""
    if dist.get_rank() == rank:
        print(f"[ RANK {rank} ]: {msg}")
        return


def setup_nccl_2(rank: int, world_size: int):
    # initialize the process group
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def setup_nccl(device: Optional[int] = None):
    # initialize the process group
    if device is not None:
        dist.init_process_group("nccl", device_id=device)
        return
    dist.init_process_group("nccl")
    return


def cleanup_nccl():
    dist.destroy_process_group()


def accelerate_setup_model(
    accelerator: "Accelerator",
    model_path: str,
):

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=accelerate2torch_type[accelerator.mixed_precision],
        low_cpu_mem_usage=True,
    )
    print_rank(dist.get_rank(), "accelerate_setup_model(): Model loaded successfully!")
    return model


def setup_model(model_path: str, tokenizer: "PreTrainedTokenizer"):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    return model


def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run


def format_metrics_to_gb(item):
    """quick function to format numbers to gigabyte and round to 4 digit precision"""
    metric_num = item / (1024**3)
    metric_num = round(metric_num, ndigits=4)
    return metric_num


def log_cuda_memory(prefix: str = ""):
    allocated = format_metrics_to_gb(cu.memory_allocated())
    reserved = format_metrics_to_gb(cu.memory_reserved())
    print_rank(
        dist.get_rank(),
        f"{prefix} Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB",
    )


def trace_handler(p):
    # sort_by_keyword = "self_" + "cuda" + "_time_total"
    # output = p.key_averages().table(sort_by=sort_by_keyword, row_limit=10)
    # output = p.key_averages().table()
    # print(output)
    print_rank(
        dist.get_rank(),
        f"Profiling step {p.step_num} finished, exporting trace to logdir {os.environ['PROFILE_LOGDIR']}",
    )
    p.export_chrome_trace(
        os.path.join(
            os.environ["PROFILE_LOGDIR"],
            f"trace_device-{dist.get_rank()}_step-{str(p.step_num)}.json",
        )
    )

    # with open(
    #    os.path.join(
    #        os.environ["PROFILE_LOGDIR"],
    #        f"memory_stats_device-{dist.get_rank()}_step-{str(p.step_num)}.json",
    #    ),
    #    "w",
    # ) as f:
    #    stats = cu.memory_stats()
    #    json.dump(stats, f, indent=4)


def pad_tensors(
    tensors1d: list[torch.Tensor], pad_idx: int | float
) -> list[torch.Tensor]:
    """
    Pad list of tensors to mamixum len
    """
    max_len = max([t.size(0) for t in tensors1d])
    padded_tensors = []
    for t in tensors1d:
        pad_size = max_len - t.size(0)
        if pad_size == 0:
            padded_tensors.append(t)
            continue

        pad_tensors = torch.full((pad_size,), pad_idx, dtype=t.dtype, device=t.device)
        padded = torch.cat([t, pad_tensors])
        padded_tensors.append(padded)

    return padded_tensors


def stack_tensors(
    tensors1d: list[torch.Tensor] | list[list[torch.Tensor]], pad_idx: int | float
) -> torch.Tensor:
    """
    Stack 1d tensors to a 2d tensor padded
    """
    if isinstance(tensors1d[0], list):
        tensors1d = listOfLists2List(tensors1d)
    padded_tensors = pad_tensors(tensors1d, pad_idx)
    shapes = set([t.size(0) for t in padded_tensors])
    assert len(shapes) == 1, (
        f"tensors1d: Not all tensors have the same shape: {shapes}!! Exiting..."
    )
    return torch.stack(padded_tensors)


def listOfLists2List(ls: list[list]) -> list:
    _l = []
    for t in ls:
        _l.extend(t)
    return _l


def gather_across_processes(local_data: Any) -> list[Any]:
    all_data = [None for _ in range(torch.distributed.get_world_size())]
    dist.all_gather_object(all_data, local_data)
    # dist.all_gather(all_data, local_data)
    return all_data


def average_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """Average a scalar across all GPUs."""
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t


def average_gather_rank(rank: int, tensor: torch.Tensor) -> torch.Tensor:
    """Average a scalar across all GPUs."""
    t = tensor.clone()
    dist.reduce(t, dst=rank, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t


def gather_tensor_to_device(
    tensor: torch.Tensor, gather_device: torch.device, dst_rank: int = 0
):
    """
    Gathers `tensor` from all ranks into a list, returns it only on `dst_rank`.

    Args:
        tensor (torch.Tensor): The tensor local to each rank.
        gather_device (torch.device): Device to move gathered tensors to.
        dst_rank (int): The rank that will receive the full list.

    Returns:
        List[torch.Tensor] on dst_rank, else None on other ranks.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Allocate list of tensors on dst_rank
    gather_list = (
        [torch.empty_like(tensor, device=tensor.device) for _ in range(world_size)]
        if rank == dst_rank
        else None
    )

    # Gather tensors from all ranks to dst_rank
    dist.gather(tensor, gather_list, dst=dst_rank)

    # Move gathered tensors to desired device (e.g., cuda:0 or cpu)
    if rank == dst_rank:
        return [t.to(gather_device) for t in gather_list]
    else:
        return None
