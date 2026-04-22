<font size="5">*Profiling and optimizing AI trainings*</font>

Copyrights reserved by Barcelona Supercomputer Center, BSC-CNS, Plaça d'Eusebi Güell, 1-3, 08034 Barcelona, Spain.

---

# Getting Started

## 1. Clone the Repository

Clone the repository and checkout the branch that matches your HPC cluster.

**MareNostrum 5 (BSC):**
```bash
git clone <repo url>
cd training-profiling-workshop
git checkout marenostrum5
```

**Leonardo (CINECA):**
```bash
git clone <repo url>
cd training-profiling-workshop
git checkout leonardo
```

## 2. Configure `exercises/env.sh`

Before running any exercise, edit `exercises/env.sh` to set paths that match your cluster environment. This file is sourced automatically by `slurm.sh`.

```bash
#!/bin/bash

# ─── SLURM Settings ───────────────────────────────────────────────────────────
export QUEUE="<slurm_qos>"          # SLURM QOS / queue name
export ACCOUNT="<project_account>"  # SLURM account / project
export PARTITION="<partition>"      # SLURM partition name

# ─── Exercise 1 (DDP) ─────────────────────────────────────────────────────────
export EX1_DATASET_PATH="<path/to/alpaca_data_cleaned.json>"
export EX1_CONTAINER_IMAGE="<path/to/ai-profiling-workshop.sif>"

# ─── Exercise 2 (DeepSpeed) ───────────────────────────────────────────────────
export EX2_DATASET_PATH="<path/to/alpaca_data_cleaned.json>"
export EX2_CONTAINER_IMAGE="<path/to/ai-profiling-workshop.sif>"

# ─── Exercise 3 (MegatronLM) ──────────────────────────────────────────────────
export EX3_DATASET_PATH="<path/to/megatron/DATA>"       # directory containing the pre-tokenised dataset
export EX3_DATASET_BIN="<relative/path/text_document>"  # relative path inside EX3_DATASET_PATH to the .bin/.idx prefix
export EX3_CONTAINER_IMAGE="<path/to/nemo.sif>"

# ─── nsys2prv (optional) ──────────────────────────────────────────────────────
export NSYS2PRV_CONTAINER_IMAGE="<path/to/ai-profiling-workshop-nsys2prv.sif>"
```

> **Note:** `EX3_DATASET_BIN` is a path *relative* to `EX3_DATASET_PATH` that points to the MegatronLM pre-processed binary prefix (without the `.bin`/`.idx` extension).

---

# Repository Structure

```
training-profiling-workshop/
├── README.md
├── exercises/
│   ├── __init__.py
│   ├── slurm.sh
│   ├── exercise_0_Communication_Tests/
│   │   └── tests.sh
│   ├── exercise_1_DDP/
│   │   ├── __init__.py
│   │   ├── ddp_config.yml
│   │   ├── slurm_nsys.sh
│   │   └── train.py
│   ├── exercise_2_DeepSpeed/
│   │   ├── __init__.py
│   │   ├── accelerate_config.yaml
│   │   ├── ds_configs/
│   │   ├── slurm_nsys.sh
│   │   └── train.py
│   ├── exercise_3_MegatronLM/
│   │   └── slurm_nsys.sh
│   └── utils/
│       ├── __init__.py
│       ├── argparsers/
│       ├── exceptions.py
│       ├── gpus_monitor.py
│       └── utils.py
├── nsys2prv/
│   ├── configs/
│   ├── nsys2prv-folder-sbatch.sh
│   └── nsys2prv-folder.sh
├── singularity-images/
│   └── def/
│       ├── ai-profiling-workshop-nsys2prv.def
│       └── ai-profiling-workshop.def
```

# How to Execute Exercises Using slurm.sh

The `slurm.sh` script in the `exercises/` directory is a wrapper for submitting SLURM jobs for all exercises. It must be run from within the `exercises/` directory.

**Basic usage:**

```bash
cd exercises
bash slurm.sh -e <EXERCISE_NUMBER> [OPTIONS] -- [EXTRA_ARGS]
```

**Key options:**
- `-e, --exercise`   : Exercise number (1, 2, or 3) (**required**)
- `-n, --nodes`      : Number of nodes (default: 1)
- `-g, --gpus-per-node`: GPUs per node (default: 4)
- `-q, --queue`      : SLURM queue/QOS (default: acc_bench)
- `-a, --account`    : Account name (default: bsc99)
- `-p, --partition`  : Partition (default: acc)
- `-h, --help`       : Show help message

**Extra arguments** (after `--`) are passed to the training script, e.g.:
- `--model NAME`              : Model name to profile (**required** — see `--models_list`)
- `--models_list`             : List available models for the selected exercise, then exit
- `--slow-dataloading`        : Enable slow dataloading example for exercise 1 (DDP)
- `--mixed-precision TYPE`    : Enable mixed precision — `bf16`, `fp16`, or `fp8` (default: disabled)
- `--micro-batch-size N`      : Set micro batch size (default: 4)
- `--gradient-accumulation-steps N` : Set gradient accumulation steps (default: 1)
- `--activation-checkpointing` : Enable activation/gradient checkpointing
- `--ds-hpz-partition N`      : DeepSpeed HPZ partition size — number of GPUs per model replica (exercise 2 only, default: 4)
- `--ds-heavy-comm`           : Enable example of heavier communication chunk sizes in DeepSpeed (exercise 2 only)
- `--ds-stage2`               : Use DeepSpeed ZeRO stage 2 instead of stage 3 (exercise 2 only)
- `--ds-no-overlap`           : Disable overlap communication in DeepSpeed (exercise 2 only)
- `--tp N`                    : Tensor parallelism degree (exercise 3 only, default: 1)
- `--pp N`                    : Pipeline parallelism degree (exercise 3 only, default: 1)
- `--cp N`                    : Context parallelism degree (exercise 3 only, default: 1)
- `--ep N`                    : Expert parallelism degree for MoE models (exercise 3 only, default: 1)
- `--global-batch-size N`     : Global batch size (exercise 3 only, default: 128)
- `--no-profile`              : Disable NSYS profiling (default: profiling enabled)
- `--nsys2prv`                : Auto-translate NSYS reports to Paraver traces after profiling (default: disabled)

**Example:**
```bash
bash slurm.sh -e 2 -- --mixed-precision --micro-batch-size 4 --gradient-accumulation-steps 16
```

# Exercise Descriptions

## Exercise 0: Communication Tests
**Goal:**
Test intra-node GPU discovery and GPU communication (intra-node and inter-node) using nvidia command `nvidia-smi`. 

**How to run:**
- Use `tests.sh` to run communication tests.

## Exercise 1: DDP (Distributed Data Parallel)
**Goal:**
Train a language model using PyTorch DDP via HuggingFace Accelerate. Focus on profiling and analyzing distributed training, memory usage, and communication overhead. Options for mixed precision, slow dataloading, and gradient accumulation.


**How to run:**
- Use `slurm.sh -e 1` with extra arguments as needed.

**Arguments to experiment with:**

| Argument | Default | Description |
|---|---|---|
| `--model NAME` | — | Model to profile (required — run `--models_list` to see available models) |
| `--micro-batch-size N` | `4` | Per-device batch size. Larger values increase GPU memory usage and throughput |
| `--gradient-accumulation-steps N` | `1` | Accumulate gradients over N steps before a weight update. Simulates a larger effective batch size |
| `--mixed-precision TYPE` | disabled | Use `bf16`, `fp16`, or `fp8` to reduce memory and speed up compute |
| `--slow-dataloading` | disabled | Artificially degrades DataLoader performance — useful to observe dataloading bottlenecks in profiles |

**Example:**
```bash
bash slurm.sh -e 1 -- --model Llama_32_3B --mixed-precision bf16 --micro-batch-size 4 --gradient-accumulation-steps 4
```

## Exercise 2: DeepSpeed ZeRO
**Goal:**
Train a language model using DeepSpeed ZeRO-2 or ZeRO-3 with hierarchical partitioning (hpZ). Explore advanced memory optimization, model sharding, and scaling. Configurable via DeepSpeed config files and extra arguments for partitioning, mixed precision, and more.


**How to run:**
- Use `slurm.sh -e 2` with DeepSpeed-specific arguments (see above).

**Arguments to experiment with:**

| Argument | Default | Description |
|---|---|---|
| `--model NAME` | — | Model to profile (required — run `--models_list` to see available models) |
| `--micro-batch-size N` | `4` | Per-device batch size per ZeRO shard |
| `--gradient-accumulation-steps N` | `1` | Accumulate gradients over N steps before a weight update |
| `--mixed-precision TYPE` | disabled | Use `bf16`, `fp16`, or `fp8` mixed precision |
| `--activation-checkpointing` | disabled | Recompute activations on backward pass to save GPU memory at the cost of extra compute |
| `--ds-stage2` | disabled (stage 3) | Switch from ZeRO-3 to ZeRO-2 partitioning — shards only gradients and optimizer states, not parameters |
| `--ds-hpz-partition N` | `4` | Number of GPUs per ZeRO partition group (hierarchical partitioning). With 32 GPUs and `--ds-hpz-partition 4`, you get 8 data-parallel replicas each sharded across 4 GPUs |
| `--ds-no-overlap` | disabled | Disable gradient/communication overlap in DeepSpeed — useful to measure the communication overhead in profiles |
| `--ds-heavy-comm` | disabled | Enable a heavier communication chunk size example — demonstrates how chunk sizing affects communication efficiency |

**Example:**
```bash
bash slurm.sh -e 2 -- --model Llama_31_8B --mixed-precision bf16 --micro-batch-size 2 --activation-checkpointing --ds-hpz-partition 4
```

## Exercise 3: MegatronLM
**Goal:**
Train a large language model using Megatron-LM, exploring tensor and pipeline parallelism, and advanced distributed strategies. Profiling and monitoring are integrated. Highly configurable for research on large-scale model training.


**How to run:**
- Use `slurm.sh -e 3` with Megatron-specific arguments (see above).

**Arguments to experiment with:**

| Argument | Default | Description |
|---|---|---|
| `--model NAME` | — | Model to profile (required — run `--models_list` to see available models) |
| `--micro-batch-size N` | `1` | Micro batch size per pipeline stage. Determines the granularity of pipeline bubbles |
| `--global-batch-size N` | `128` | Total batch size across all GPUs. Must be divisible by `micro-batch-size × data-parallel-degree` |
| `--mixed-precision TYPE` | disabled | Use `bf16`, `fp16`, or `fp8` (TransformerEngine FP8) mixed precision |
| `--activation-checkpointing` | disabled | Enable activation recomputation to save memory at the cost of extra compute |
| `--tp N` | Tensor parallelism degree — splits attention heads and FFN layers across N GPUs (intra-node) |
| `--pp N` | Pipeline parallelism degree — partitions model layers across N pipeline stages (inter-node) |
| `--cp N` | Context parallelism degree — splits the sequence dimension across N GPUs for long-context training |
| `--ep N` | Expert parallelism degree for Mixture-of-Experts (MoE) models only |

> **Constraint:** `TP × PP × CP × DP = total GPUs`. For example with 16 GPUs: `TP=4, PP=2, CP=1` → `DP=2`.

**Example:**
```bash
bash slurm.sh -n 2 -g 4 -e 3 -- --model Llama_31_8B --mixed-precision bf16 --micro-batch-size 2 --global-batch-size 16 --tp 4 --pp 2
```

---