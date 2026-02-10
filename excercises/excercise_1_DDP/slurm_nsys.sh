#!/bin/bash

#SBATCH --job-name=excercise_1_ddp_nsys
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

# ============================================================
# NSYS Profiling for Accelerate DDP Training
# ============================================================
# This script profiles distributed training with NVIDIA Nsight Systems
# Each GPU process gets its own nsys profile for detailed analysis
# ============================================================

# Load required modules
module purge
module load singularity
module load cuda/12.6  # Ensure nsys is available

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# NCCL configuration for multi-node
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

#cd "$EXCERCISE_DIR" || { echo "Error: Exercise directory not found: $EXCERCISE_DIR"; exit 1; }
export ACCELERATE_CONFIG_FILE="$EXCERCISE_DIR/ddp_config.yml"

export PROFILER_PREFIX_PATH="$EXCERCISE_DIR"
export GPUS_MONITOR_PREFIX_PATH="$EXCERCISE_DIR"
export LOGLEVEL=INFO
export TOKENIZERS_PARALLELISM=false

# Dataset and model paths
DATASET_PATH="../data/text2text/instructions/alpaca-cleaned/alpaca_data_cleaned.json"
MODEL_PATH="/gpfs/scratch/bsc99/ai_operations/models_registry/models_registry/Llama-3.1-1B"
CONTAINER_IMAGE="../singularity-images/ai-profiling-workshop.sif"

which python
which nsys

# Get head node for torchrun rendezvous
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=$NUM_NODES --ntasks=1 -w "$head_node" hostname --ip-address)
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

echo "$ECHO_PREFIX Node IP: $head_node_ip"
export LOGLEVEL=DEBUG

tmp_config="$ACCELERATE_CONFIG_FILE-$RANDOM_PREFIX"

# Ensure tmp_config is always deleted on exit (success, failure, or signal)
cleanup() {
    if [[ -f "$tmp_config" ]]; then
        rm -f "$tmp_config"
        echo "$ECHO_PREFIX Cleaned up temporary config: $tmp_config"
    fi
}
trap cleanup EXIT

cp "$ACCELERATE_CONFIG_FILE" "$tmp_config"

num_processes=$((NUM_NODES*NUM_GPUS))
sed -i "s/main_process_ip: ''/main_process_ip: $head_node_ip/g" "$tmp_config"
sed -i "s/num_machines: 0/num_machines: $NUM_NODES/g" "$tmp_config"
sed -i "s/num_processes: 0/num_processes: $num_processes/g" "$tmp_config"


gpu_monitor_command="python -m utils.gpus_monitor"

singularity_prefix="singularity exec --network host --nv --bind /apps:/apps $CONTAINER_IMAGE"

python_modulde="excercise_1_DDP.train"
train_command="$singularity_prefix accelerate launch \
    --config_file $tmp_config \
    --rdzv_backend=c10d \
    --machine_rank $machine_rank \
    -m  $python_modulde \
        --data-path $DATASET_PATH \
        --model-path $MODEL_PATH \
        --epochs 1 \
        --no-validation \
        --profile \
        --batch-size 4 \
        --gradient-accumulation-steps 4 \
"

NSYS_OUTPUT_DIR="$PROFILER_PREFIX_PATH/profiler/$SLURM_JOB_ID-nsys"
export GPUS_MONITOR_PREFIX_PATH="$PROFILER_PREFIX_PATH"
export TRAINING_ARGUMENTS_FILE="$NSYS_OUTPUT_DIR/training_arguments.json"
mkdir -p "$NSYS_OUTPUT_DIR"

# ============================================================
# NSYS Configuration
# ============================================================
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
# ============================================================

# NSYS profiling options
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
    --sample=cpu \
    --backtrace=dwarf \
    --stats=true \
    --force-overwrite=true \
    --output=${NSYS_OUTPUT_DIR}/profile_node%q{SLURM_NODEID}_rank%q{SLURM_LOCALID} \
"

train_command="nsys profile $NSYS_OPTS $train_command"

echo "$ECHO_PREFIX ============================================================"
echo "$ECHO_PREFIX Starting NSYS profiling for Accelerate DDP..."
echo "$ECHO_PREFIX Output directory: $NSYS_OUTPUT_DIR"
echo "$ECHO_PREFIX ============================================================"


srun --export=ALL bash -c "
    # Start monitoring in background
    $gpu_monitor_command &
    monitor_pid=\$!

    # Optional: give the monitor time to initialize
    sleep 5

    # Run training in foreground (this blocks until done)
    $train_command

    kill -SIGTERM \"\$monitor_pid\"

    # Wait for the monitor to clean up and exit
    wait \"\$monitor_pid\"
"

if [ $? -ne 0 ]; then
    echo "$ECHO_PREFIX Python script failed. Exiting."
    exit 1
fi

echo "$ECHO_PREFIX Python script succeeded."
echo "$ECHO_PREFIX ============================================================"
echo "$ECHO_PREFIX NSYS profiling complete!"
echo "$ECHO_PREFIX Output files: $NSYS_OUTPUT_DIR/"
echo "$ECHO_PREFIX "
echo "$ECHO_PREFIX To analyze:"
echo "$ECHO_PREFIX   1. Download .nsys-rep files to local machine"
echo "$ECHO_PREFIX   2. Open with: nsys-ui <file>.nsys-rep"
echo "$ECHO_PREFIX   3. Or generate stats: nsys stats <file>.nsys-rep"
echo "$ECHO_PREFIX ============================================================"