#!/bin/bash

#SBATCH --job-name=2_fsdp_deepspeed_nsys
#SBATCH --output={{LOG_OUT}}
#SBATCH --error={{LOG_ERR}}
#SBATCH --nodes={{NUM_NODES}}
#SBATCH --gres=gpu:{{NUM_GPUS}}
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --account={{ACCOUNT}}
#SBATCH --qos={{QUEUE}}
#SBATCH --partition={{PARTITION}}


# ============================================================================
# NSYS Profiling for DeepSpeed ZeRO-3 Training
# ============================================================================
#
# This script profiles DeepSpeed distributed training with NVIDIA Nsight Systems.
# Each GPU process gets its own nsys profile for detailed analysis.
#
# KEY CONCEPT - Understanding hpZ:
# --------------------------------
# Example below:
# With ZeRO-3 + hpZ, you control how many GPUs share sharded parameters:
#
#   32 GPUs total, hpz_partition_size=8:
#   ├── Partition Group 0: GPUs 0-7   (shard model across 8 GPUs)
#   ├── Partition Group 1: GPUs 8-15  (shard model across 8 GPUs)
#   ├── Partition Group 2: GPUs 16-23 (shard model across 8 GPUs)
#   └── Partition Group 3: GPUs 24-31 (shard model across 8 GPUs)
#
# Result: 4 data parallel replicas, each holding 1/8 of the model per GPU
#
# NVTX Markers in trace:
#   - epoch_N      : Full epoch boundary
#   - batch_N      : Individual batch processing
#   - forward      : Model forward pass
#   - backward     : Gradient computation
#   - optimizer_step : Weight update
#
# ============================================================================

# Load required modules
module purge
module load singularity
module load cuda/12.6  # Ensure nsys is available

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# =============================================
# NCCL Configuration for Multi-Node InfiniBand
# =============================================
export NCCL_NET=IB
export NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT

export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export PROFILER_PREFIX_PATH="$EXCERCISE_DIR"
export GPUS_MONITOR_PREFIX_PATH="$EXCERCISE_DIR"
export LOGLEVEL=INFO
export TOKENIZERS_PARALLELISM=false

export ACCELERATE_CONFIG_FILE="$EXCERCISE_DIR/accelerate_config.yaml"
#export DS_CONFIG_FILE="$EXCERCISE_DIR/ds_config.json"
export DS_CONFIG_FILE="$EXCERCISE_DIR/ds_config_full-precision.json"

# Dataset and model paths
DATASET_PATH="../data/text2text/instructions/alpaca-cleaned/alpaca_data_cleaned.json"
MODEL_PATH="/apps/ai-hub/models/text-models/meta-llama/Llama-3.1/Llama-3.1-8B"
MODEL_PATH="/apps/ai-hub/models/text-models/meta-llama/Llama-3.1/Llama-3.1-1B"
CONTAINER_IMAGE="../singularity-images/ai-profiling-workshop.sif"

# DeepSpeed specific vars
export HPZ_PARTITION_SIZE=${HPZ_PARTITION_SIZE:-4} # Number of gpus per model replica
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
MIXED_PRECISION=${MIXED_PRECISION:-0}
ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING:-0}

if [[ $MIXED_PRECISION -eq 1 ]]; then 
    export DS_CONFIG_FILE="$EXCERCISE_DIR/ds_config_mixed-precision_no-activation-checkpointing.json" 
    if [[ $ACTIVATION_CHECKPOINTING -eq 1 ]]; then
        export DS_CONFIG_FILE="$EXCERCISE_DIR/ds_config_mixed-precision.json"
    fi
fi

which python
which nsys

# Fix for:
# df: .triton/autotune: No such file or directory
# FileNotFoundError: [Errno 2] No such file or directory: '.triton/autotune/Fp16Matmul_2d_kernel.pickle.tmp'
export TRITON_CACHE_DIR="./.cache/triton_cache_${SLURM_JOB_ID}_node-${SLURM_NODEID}"
mkdir -p "$TRITON_CACHE_DIR"
export TRITON_BACKEND_CACHE_DIR="$TRITON_CACHE_DIR"

# ============================================================================
# Node Discovery and Rank Assignment
# ============================================================================
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

this_node=$(hostname)
machine_rank=-1
for i in "${!nodes_array[@]}"; do
  if [[ "${nodes_array[i]}" == "$this_node" ]]; then
    machine_rank=$i
    break
  fi
done

ECHO_PREFIX="[Node $machine_rank]"
echo "$ECHO_PREFIX Head Node IP: $head_node_ip"
echo "$ECHO_PREFIX NUM_NODES: $NUM_NODES"
echo "$ECHO_PREFIX NUM_GPUS: $NUM_GPUS"
echo "$ECHO_PREFIX SLURM_JOB_ID: $SLURM_JOB_ID"

if [[ $machine_rank -eq -1 ]]; then
  echo "$ECHO_PREFIX Error: This node ($this_node) not found in the SLURM node list!"
  exit 1
fi

echo "$ECHO_PREFIX Node $this_node assigned machine_rank=$machine_rank"

export LOGLEVEL=DEBUG

# ============================================================================
# DeepSpeed Configuration Setup
# ============================================================================
# Create temporary copies of config files with substituted values
tmp_accelerate_config="$ACCELERATE_CONFIG_FILE-$RANDOM_PREFIX"
tmp_ds_config="$DS_CONFIG_FILE-$RANDOM_PREFIX"
cp "$ACCELERATE_CONFIG_FILE" "$tmp_accelerate_config"
cp "$DS_CONFIG_FILE" "$tmp_ds_config"

# Ensure tmp_config is always deleted on exit (success, failure, or signal)
cleanup() {
    if [[ -f "$tmp_accelerate_config" ]]; then
        rm -f "$tmp_accelerate_config"
        echo "$ECHO_PREFIX Cleaned up temporary accelerate config: $tmp_accelerate_config"
    fi
    if [[ -f "$tmp_ds_config" ]]; then
        rm -f "$tmp_ds_config"
        echo "$ECHO_PREFIX Cleaned up temporary deepspeed config: $tmp_ds_config"
    fi
}
trap cleanup EXIT

num_processes=$((NUM_NODES * NUM_GPUS))

# Update Accelerate config placeholders
sed -i "s/{{MASTER_IP}}/$head_node_ip/g" "$tmp_accelerate_config"
sed -i "s/{{NUM_NODES}}/$NUM_NODES/g" "$tmp_accelerate_config"
sed -i "s/{{NUM_GPUS}}/$num_processes/g" "$tmp_accelerate_config"

# Update DeepSpeed config path in Accelerate config
sed -i "s|{{path to ds_config.json}}|$tmp_ds_config|g" "$tmp_accelerate_config"

# Update hpZ partition size if specified via environment variable
if [[ -n "$HPZ_PARTITION_SIZE" ]]; then
    sed -i "s/\"zero_hpz_partition_size\": \"{{HPZ_PARTITION_SIZE}}\"/\"zero_hpz_partition_size\": $HPZ_PARTITION_SIZE/g" "$tmp_ds_config"
    echo "$ECHO_PREFIX Using hpZ partition size: $HPZ_PARTITION_SIZE"
fi


echo "$ECHO_PREFIX =============================================="
echo "$ECHO_PREFIX DeepSpeed ZeRO-3 NSYS Profiling Configuration"
echo "$ECHO_PREFIX =============================================="
echo "$ECHO_PREFIX Dataset Path: $DATASET_PATH"
echo "$ECHO_PREFIX Model Path: $MODEL_PATH"
echo "$ECHO_PREFIX Nodes: $NUM_NODES"
echo "$ECHO_PREFIX GPUs per Node: $NUM_GPUS"
echo "$ECHO_PREFIX Total GPUs: $num_processes"
echo "$ECHO_PREFIX hpZ Partition Size: $HPZ_PARTITION_SIZE"
echo "$ECHO_PREFIX Data Parallel Replicas: $((num_processes / HPZ_PARTITION_SIZE))"
echo "$ECHO_PREFIX =============================================="

# ============================================================================
# Training Command
# ============================================================================
#--bind /dev/infiniband --bind /dev/gdrdrv --bind /etc/infiniband --bind /dev/shm \
singularity_prefix="singularity exec --network host --nv \
    --bind /apps:/apps \
    $CONTAINER_IMAGE"

gpu_monitor_command="$singularity_prefix python -m utils.gpus_monitor"

python_module="excercise_2_DeepSpeed.train"
train_command="$singularity_prefix accelerate launch \
    --config_file $tmp_accelerate_config \
    --rdzv_backend=c10d \
    --machine_rank $machine_rank \
    -m $python_module \
        --data-path $DATASET_PATH \
        --model-path $MODEL_PATH \
        --epochs 1 \
        --no-validation \
        --profile \
        --data-sample 2000 \
        --dataloader-num-workers 8 \
        --batch-size $MICRO_BATCH_SIZE \
        --gradient-accumulation-steps $GRADIENT_ACCUMULATION_STEPS \
        --deepspeed-config $DS_CONFIG_FILE"


# ============================================================================
# NSYS Output Directory
# ============================================================================
NSYS_OUTPUT_DIR="$PROFILER_PREFIX_PATH/profiler/$SLURM_JOB_ID-nsys"
export TRAINING_ARGUMENTS_FILE="$NSYS_OUTPUT_DIR/training_arguments.json"
mkdir -p "$NSYS_OUTPUT_DIR"

# ============================================================================
# NSYS Configuration
# ============================================================================
# Key options explained:
#   --trace=cuda,nvtx,osrt,cudnn,cublas : Trace GPU kernels, NVTX markers, OS runtime, cuDNN, cuBLAS
#   --cuda-memory-usage=true            : Track CUDA memory allocations
#   --gpuctxsw=true                     : Track GPU context switches
#   --gpu-metrics-device=all            : Collect GPU metrics (SM utilization, memory throughput, etc.)
#   --gpu-metrics-frequency=10000       : Sample GPU metrics at 10kHz for fine granularity
#   --capture-range=cudaProfilerApi     : Use cudaProfiler start/stop for precise capture
#   --capture-range-end=stop            : End capture when cudaProfilerStop is called
#   --sample=cpu                        : CPU sampling for host-side bottlenecks
#   --backtrace=dwarf                   : Detailed backtraces for CPU samples
#   --stats=true                        : Generate summary statistics
# ============================================================================

# Profiler schedule: skip_first + wait + warmup = start of active window
export PROFILE_SKIP_FIRST=10
export PROFILE_WAIT=1
export PROFILE_WARMUP=5
export PROFILE_STEPS_INTERVAL=20

NSYS_OPTS=" \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --gpuctxsw=true \
    --gpu-metrics-device=all \
    --gpu-metrics-frequency=10000 \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --cudabacktrace=kernel \
    --stats=true \
    --force-overwrite=true \
    --output=${NSYS_OUTPUT_DIR}/profile_node%q{SLURM_NODEID}_rank%q{SLURM_LOCALID} \
"

train_command="nsys profile $NSYS_OPTS $train_command"

echo "train_command: $train_command"

echo "$ECHO_PREFIX ============================================================"
echo "$ECHO_PREFIX Starting NSYS profiling for DeepSpeed ZeRO-3..."
echo "$ECHO_PREFIX Output directory: $NSYS_OUTPUT_DIR"
echo "$ECHO_PREFIX ============================================================"

# ============================================================================
# Launch Training with GPU Monitoring
# ============================================================================
srun --export=ALL bash -c "
    # Start GPU monitoring in background
    $gpu_monitor_command &
    monitor_pid=\$!

    # Allow monitor to initialize
    sleep 5

    # Run training with nsys profiling (blocks until complete)
    $train_command

    # Cleanup monitor
    kill -SIGTERM \"\$monitor_pid\"
    wait \"\$monitor_pid\"
"

# ============================================================================
# Cleanup
# ============================================================================

if [ $? -ne 0 ]; then
    echo "$ECHO_PREFIX Training failed. Exiting."
    exit 1
fi

echo "$ECHO_PREFIX Training completed successfully."
echo "$ECHO_PREFIX ============================================================"
echo "$ECHO_PREFIX NSYS profiling complete!"
echo "$ECHO_PREFIX Output files: $NSYS_OUTPUT_DIR/"
echo "$ECHO_PREFIX "
echo "$ECHO_PREFIX To analyze:"
echo "$ECHO_PREFIX   1. Download .nsys-rep files to local machine"
echo "$ECHO_PREFIX   2. Open with: nsys-ui <file>.nsys-rep"
echo "$ECHO_PREFIX   3. Or generate stats: nsys stats <file>.nsys-rep"
echo "$ECHO_PREFIX ============================================================"
