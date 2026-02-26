#!/bin/bash

#SBATCH --job-name=1_ddp_nsys
#SBATCH --output={{LOG_OUT}}
#SBATCH --error={{LOG_ERR}}
#SBATCH --nodes={{NUM_NODES}}
#SBATCH --gres=gpu:{{NUM_GPUS}}
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --account={{ACCOUNT}}
##SBATCH --qos={{QUEUE}}
#SBATCH --partition={{PARTITION}}

# ============================================================
# NSYS Profiling for Accelerate DDP Training
# ============================================================
# This script profiles distributed training with NVIDIA Nsight Systems
# Each GPU process gets its own nsys profile for detailed analysis
# ============================================================

# Load required modules
module purge
module load cuda/12.6  # Ensure nsys is available

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# =============================================
# NCCL Configuration for Multi-Node InfiniBand
# =============================================
export NCCL_DEBUG=INFO
export LD_PRELOAD=""

export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

#cd "$EXERCISE_DIR" || { echo "Error: Exercise directory not found: $EXERCISE_DIR"; exit 1; }
export ACCELERATE_CONFIG_FILE="$EXERCISE_DIR/ddp_config.yml"

export LOGLEVEL=INFO
export TOKENIZERS_PARALLELISM=false

# Dataset and model paths
DATASET_PATH="/leonardo_work/tra26_minwinsc/datasets/alpaca-cleaned/alpaca_data_cleaned.json"
MODEL_PATH="/leonardo_work/tra26_minwinsc/Llama-3.1-1B"
CONTAINER_IMAGE="/leonardo_work/tra26_minwinsc/bsc-containers/ai-profiling-workshop.sif"

SLOW_DATALOADING=${SLOW_DATALOADING:-0} # Boolean flag to enable slow dataloading (for testing bottlenecks)
MIXED_PRECISION=${MIXED_PRECISION:-0} # Boolean flag to enable mixed precision (e.g., bf16)
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}

NO_PROFILE=${NO_PROFILE:-0} # Boolean flag to disable profiling (for testing without nsys overhead)
NSYS2PRV=${NSYS2PRV:-0} # Boolean flag to enable nsys2prv translation after profiling
#which python
echo "NSYS path: $(which nsys)"

# ============================================================================
# Node Discovery and Rank Assignment
# ============================================================================
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=$NUM_NODES --ntasks=1 -w "$head_node" hostname --ip-address)
this_node=$(hostname)

for i in "${!nodes_array[@]}"; do
  nodes_array[$i]="${nodes_array[$i]}.leonardo.local"
  echo "node $i: ${nodes_array[i]}"
done


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
ABSOLUTE_EXERCISE_DIR="$(realpath "$EXERCISE_DIR")"
singularity_prefix="singularity exec --network host --nv \
	--bind /leonardo_work \
	--bind /leonardo \
    --bind "$ABSOLUTE_EXERCISE_DIR":"$ABSOLUTE_EXERCISE_DIR" \
	$CONTAINER_IMAGE"

gpu_monitor_command="$singularity_prefix python -m utils.gpus_monitor"

python_modulde="exercise_1_DDP.train"
train_command="$singularity_prefix accelerate launch \
    --config_file $tmp_config \
    --rdzv_backend=c10d \
    --machine_rank $machine_rank \
    -m  $python_modulde \
        --data-path $DATASET_PATH \
        --model-path $MODEL_PATH \
        --epochs 1 \
        --data-sample 5000 \
        --no-validation \
        --profile \
        --batch-size $MICRO_BATCH_SIZE \
        --gradient-accumulation-steps $GRADIENT_ACCUMULATION_STEPS"

TRAIN_CLI_ARGS=""
if [[ $SLOW_DATALOADING -eq 1 ]]; then
        TRAIN_CLI_ARGS="$TRAIN_CLI_ARGS --slow-dataloading"
fi
if [[ $MIXED_PRECISION -eq 1 ]]; then 
    TRAIN_CLI_ARGS="$TRAIN_CLI_ARGS --mixed-precision bf16"
fi

train_command="$train_command $TRAIN_CLI_ARGS"

# ============================================================================
# NSYS Output Directory
# ============================================================================
MODEL_NAME=$(basename "$MODEL_PATH")
export PROFILER_PREFIX_PATH="$EXERCISE_DIR/profiler/\
$MODEL_NAME-$SLURM_JOB_ID-\
n$SLURM_NNODES-\
g4-\
mbs$MICRO_BATCH_SIZE-\
gas${GRADIENT_ACCUMULATION_STEPS}-\
mixed${MIXED_PRECISION}-\
slow${SLOW_DATALOADING}"
export GPUS_MONITOR_PREFIX_PATH="$PROFILER_PREFIX_PATH"
NSYS_OUTPUT_DIR="$PROFILER_PREFIX_PATH/nsys"
if [ $NO_PROFILE -eq 1 ]; then
    echo "$ECHO_PREFIX Profiling is disabled. NSYS output will not be generated."
    NSYS_OUTPUT_DIR="$PROFILER_PREFIX_PATH/no-nsys"
fi
export TRAINING_ARGUMENTS_FILE="$NSYS_OUTPUT_DIR/training_arguments.json"
mkdir -p "$NSYS_OUTPUT_DIR"

# ============================================================================
# NSYS Configuration
# ============================================================================
# Recommended NSYS options for ML training profiling:
#   --trace=cuda,nvtx,osrt,cudnn,ucx,nccl,cublas : Trace GPU kernels, NVTX markers, OS runtime, cuDNN, UCX, NCCL, cuBLAS
#   --force-overwrite true              : Overwrite existing profile files
#   --cuda-memory-usage=true            : Track CUDA memory allocations and usage
#   --gpuctxsw=true                     : Track GPU context switches (optional, rarely a bottleneck)
#   --gpu-metrics-devices=all           : Collect GPU metrics (SM utilization, memory throughput, etc.)
#   --gpu-metrics-frequency=10000       : Sample GPU metrics at 10kHz for fine granularity
#   --capture-range=cudaProfilerApi     : Use cudaProfiler start/stop for precise capture (requires NVTX markers in code)
#   --capture-range-end=stop            : End capture when cudaProfilerStop is called
#   --sample=cpu                        : CPU sampling for host-side bottlenecks
#   --backtrace=dwarf                   : Detailed backtraces for CPU samples
#   --cudabacktrace=kernel              : Collect backtraces for CUDA kernel launches
#   --stats=true                        : Generate summary statistics
#   --export=sqlite                     : Export results in SQLite format for advanced analysis
#   --output=<path>                     : Set output file path (already used)
#
# ============================================================================

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
    --gpu-metrics-devices=all \
    --gpu-metrics-frequency=10000 \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --stats=true \
    --output=${NSYS_OUTPUT_DIR}/profile_node%q{SLURM_NODEID} \
"

if [ $NO_PROFILE -eq 0 ]; then
    train_command="nsys profile $NSYS_OPTS $train_command"
fi

echo "train_command: $train_command"

echo "$ECHO_PREFIX ============================================================"
echo "$ECHO_PREFIX Starting NSYS profiling for Accelerate DDP..."
echo "$ECHO_PREFIX Output directory: $NSYS_OUTPUT_DIR"
echo "$ECHO_PREFIX ============================================================"


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

    # Wait for the monitor to clean up and exit
    wait \"\$monitor_pid\"
"

# ============================================================================
# Cleanup
# ============================================================================

if [ $? -ne 0 ]; then
    echo "$ECHO_PREFIX Training failed. Exiting."
    exit 1
fi

echo "$ECHO_PREFIX Training Training."
echo "$ECHO_PREFIX ============================================================"
echo "$ECHO_PREFIX NSYS profiling complete!"
echo "$ECHO_PREFIX Output files: $NSYS_OUTPUT_DIR/"
echo "$ECHO_PREFIX "
echo "$ECHO_PREFIX To analyze:"
echo "$ECHO_PREFIX   1. Download .nsys-rep files to local machine"
echo "$ECHO_PREFIX   2. Open with: nsys-ui <file>.nsys-rep"
echo "$ECHO_PREFIX   3. Or generate stats: nsys stats <file>.nsys-rep"
echo "$ECHO_PREFIX ============================================================"

if [ $NSYS2PRV -eq 1 ]; then
    experiment_path="$PROFILER_PREFIX_PATH"
    experiment_name=$(basename "$experiment_path")
    echo "$ECHO_PREFIX Generating Paraver traces from NSYS profiles with name $experiment_name..."
    bash "../nsys2prv/nsys2prv-folder.sh" "$NSYS_OUTPUT_DIR" -n "$experiment_name"
fi