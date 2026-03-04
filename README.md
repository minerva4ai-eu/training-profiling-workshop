
<font size="6">MINERVA: AI Winter School 2026</font>

<font size="5">*Profiling and optimizing AI trainings*</font>

This repo consists educational material, created with the purpose of serving the MINERVA: AI Winter School 2026.

In this workshop we will be talking in detail about training workload profiling and its importance in optimization.

Copyrights reserved by Barcelona Supercomputer Center, BSC-CNS, Plaça d'Eusebi Güell, 1-3, 08034 Barcelona, Spain.


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
- `--slow-dataloading`: Enable slow dataloading example for exercise 1 (DDP)
- `--mixed-precision` : Enable mixed precision (bf16)
- `--micro-batch-size N` : Set micro batch size
- `--gradient-accumulation-steps N` : Set gradient accumulation steps
- `--activation-checkpointing` : Enable activation/gradient checkpointing
- `--ds-hpz-partition N` : DeepSpeed HPZ partition size (exercise 2 only)
- `--ds-heavy-comm`: Enable example of heavier communication chuck sizes in DeepSpeed (exercise 2 only)
- `--ds-stage2` : Use DeepSpeed stage 2  (exercise 2 only)
- `--ds-no-overlap`: Disable overlap communication in DeepSpeed (exercise 2 only)
- `--tp N` : Tensor parallelism (exercise 3 only)
- `--pp N` : Pipeline parallelism (exercise 3 only)
- `--global-batch-size N`: Set global batch size to N, for exercise 3 (MegatronLM) only
- `--no-profile` : Disable profiling with NSYS

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

**Prepare**
- Make sure to set the following environment variables in exercise_2_DeepSpeed/slurm_nsys.sh
```
# Dataset and model paths
DATASET_PATH="<set path to dataset here>" 
MODEL_PATH="<set path to model here>"
CONTAINER_IMAGE="<set path to container image here>"
```

**How to run:**
- Use `slurm.sh -e 1` with extra arguments as needed.

## Exercise 2: DeepSpeed ZeRO
**Goal:**
Train a language model using DeepSpeed ZeRO-2 or ZeRO-3 with hierarchical partitioning (hpZ). Explore advanced memory optimization, model sharding, and scaling. Configurable via DeepSpeed config files and extra arguments for partitioning, mixed precision, and more.

**Prepare**
- Make sure to set the following environment variables in exercise_2_DeepSpeed/slurm_nsys.sh
```
# Dataset and model paths
DATASET_PATH="<set path to dataset here>" 
MODEL_PATH="<set path to model here>"
CONTAINER_IMAGE="<set path to container image here>"
```

**How to run:**
- Use `slurm.sh -e 2` with DeepSpeed-specific arguments (see above).

## Exercise 3: MegatronLM
**Goal:**
Train a large language model using Megatron-LM, exploring tensor and pipeline parallelism, and advanced distributed strategies. Profiling and monitoring are integrated. Highly configurable for research on large-scale model training.

**Prepare**
- Make sure to set the following environment variables in exercise_2_DeepSpeed/slurm_nsys.sh
```
# === Singularity Image Path ===
export PATH_SINGULARITY="<set path to container image here>"

# === Host Folder Bind Mount Setup ===
export PATH_TOKENIZER="<set path to tokenizer here>"
export PATH_MODEL="<set path to model here>"
export PATH_DATA="<set path to data here>"
export PATH_TO_BIN="<set path to binary here>" # path inside the data folder containing .bin & .idx
```

**How to run:**
- Use `slurm.sh -e 3` with Megatron-specific arguments (see above).

---