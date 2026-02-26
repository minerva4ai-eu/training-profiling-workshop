#!/bin/bash
#SBATCH --job-name=3_megatronLM_nsys
#SBATCH --output={{LOG_OUT}}
#SBATCH --error={{LOG_ERR}}
#SBATCH --nodes={{NUM_NODES}}
#SBATCH --gres=gpu:{{NUM_GPUS}}
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --account={{ACCOUNT}}
##SBATCH --qos={{QUEUE}}
#SBATCH --partition={{PARTITION}}

module purge
module load cuda/12.6

export NUMEXPR_MAX_THREADS=256
export NCCL_P2P_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1 
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
#export LD_PRELOAD=""

# === Singularity Image Path ===
export PATH_SINGULARITY="/leonardo_work/tra26_minwinsc/bsc-containers/nemo_25.07.sif"

# === Host Folder Bind Mount Setup ===
export PATH_TOKENIZER="/leonardo_work/tra26_minwinsc/models/Mistral-7B-v0.1"
export PATH_MODEL="/leonardo_work/tra26_minwinsc/models/Mistral-7B-v0.1"
export PATH_DATA="/leonardo_work/tra26_minwinsc"
export PATH_TO_BIN="FW/fineweb-10BT_text_document" # path inside the data folder
export PATH_RESULTS="$EXERCISE_DIR/results"
export PATH_LOGS="$EXERCISE_DIR/logs-megatronlm"
export PATH_CACHE="$EXERCISE_DIR/.cache-megatronlm"

mkdir -p "$PATH_RESULTS" "$PATH_LOGS" "$PATH_CACHE"

# === Megatron Config Defaults ===
MODEL_SIZE="mistral7b"
TP="${TP:-4}"
PP="${PP:-1}"
CP="${CP:-1}"
#EP="${EP:-2}"
MBS="${MICRO_BATCH_SIZE:-4}"
GBS="${GLOBAL_BATCH_SIZE:-128}" # -> 
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
MAX_POSITION_EMBEDDINGS=$SEQ_LENGTH
TOTAL_ITERS="${TOTAL_ITERS:-10}" #####################################original 100 iter, set to 10 to speed up 
TOKENIZER_TYPE="${TOKENIZER_TYPE:-HuggingFaceTokenizer}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-tokenizer.model}" ##########################################################################tokenizer
TE_FP8="${TE_FP8:-0}" # 0 for bf16
FSDP="${FSDP:-0}"
RECOMPUTE="${RECOMPUTE:-0}"
OPTIMIZER="${OPTIMIZER:-adam}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
EVAL_ITERS="${EVAL_ITERS:--1}"
CKPT_FORMAT="${CKPT_FORMAT:-torch}"

NO_PROFILE="${NO_PROFILE:-0}" # Boolean flag to disable profiling (for testing without nsys overhead)
NSYS2PRV="${NSYS2PRV:-0}" # Boolean flag to enable nsys2prv translation after profiling
#export NCCL_DEBUG=INFO

echo "=== Configuration ==="
echo "MODEL_SIZE: $MODEL_SIZE"
echo "Nodes: $SLURM_NNODES"
echo "TP: $TP"
echo "PP: $PP"
echo "CP: $CP"
#echo "EP: $EP"
echo "MBS: $MBS"
echo "GBS: $GBS"


# === Model Hyperparameters ===
if [[ $MODEL_SIZE -eq "mistral7b" ]]; then # based on
   # Core architecture
    HIDDEN_SIZE=4096
    FFN_HIDDEN_SIZE=14336
    NUM_LAYERS=32
    NUM_HEADS=32
    NUM_KV_HEADS=8
    MAX_POSITION_EMBEDDINGS=4096
    SEQ_LENGTH=4096
    INIT_METHOD_STD=0.005
    ROTARY_BASE=1000000
fi
GROUP_SIZE=$(( NUM_HEADS / NUM_KV_HEADS ))
NUM_GROUPS=$(( NUM_HEADS / GROUP_SIZE ))


# === Adjust TP if FSDP is set ===
if [[ "$FSDP" -eq 1 && "$TP" -gt 1 ]]; then
    echo "FSDP and TP are not compatible. Setting TP=1."
    export TP=1
fi

# Distributed args
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
nodes_array=("${nodes[@]}")
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
export NNODES=${#nodes_array[@]}
export GPUS_PER_NODE=4
export WORLD_SIZE=$NNODES*$GPUS_PER_NODE
echo "###NNODES:$NNODES\n"
 
# === Build Argument Strings ===

export DISTRIBUTED_ARGS="--rdzv_id=$RANDOM \
 --rdzv_backend=c10d \
 --rdzv_endpoint=$head_node_ip:29505 \
 --nnodes=$NNODES \
 --nproc_per_node=$GPUS_PER_NODE"

export GPT_ARGS="\
	--tensor-model-parallel-size ${TP} \
	--sequence-parallel \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
	--use-distributed-optimizer \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --num-attention-heads ${NUM_HEADS} \
    --seq-length ${SEQ_LENGTH} \
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
	--rotary-base "${ROTARY_BASE:-500000}" \
    --no-position-embedding \
    --swiglu \
    --disable-bias-linear \
    --init-method-std "${INIT_METHOD_STD:-0.02}" \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --train-iters ${TOTAL_ITERS} \
    --no-async-tensor-model-parallel-allreduce \
    --bf16 \
	--use-flash-attn \
    --no-masked-softmax-fusion \
    --group-query-attention \
    --num-query-groups ${NUM_GROUPS}"

export TRAIN_ARGS=" \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --lr-warmup-fraction 0.01 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --weight-decay 1.0e-1 \
    --clip-grad 1.0"

export DATA_ARGS=" \
    --tokenizer-type ${TOKENIZER_TYPE} \
    --tokenizer-model /tokenizer/ \
    --data-path /data/${PATH_TO_BIN} \
    --dataloader-type cyclic \
    --num-workers 8 \
    --data-cache-path /cache"

export OUTPUT_ARGS=" \
    --log-interval ${LOG_INTERVAL} \
    --eval-interval ${EVAL_INTERVAL} \
    --eval-iters ${EVAL_ITERS} \
    --log-throughput \
    --no-save-optim \
    --no-save-rng \
    --save-interval ${SAVE_INTERVAL} \
    --ckpt-format ${CKPT_FORMAT} \
    --save /results"

export EXTRA_ARGS=" \
	--distributed-backend nccl \
	--overlap-grad-reduce \
    --overlap-param-gather"

if [[ "$RECOMPUTE" -eq 1 ]]; then
    EXTRA_ARGS+=" \
	--recompute-num-layers ${NUM_LAYERS} \ 
	--recompute-granularity full \
	--recompute-method block"
fi


if [[ "$FSDP" -eq 1 ]]; then
    EXTRA_ARGS+=" \
	--use-torch-fsdp2"
fi

if [[ "$OPTIMIZER" == "adam" ]]; then
    EXTRA_ARGS+=" \
	--optimizer adam \
	--adam-beta1 0.9 \
	--adam-beta2 0.95"
else
    EXTRA_ARGS+=" \
	--optimizer sgd"
fi


# Default to no MoE
MOE=${MOE:-0}

#	--expert-model-parallel-size ${EP} \
if [[ "$MOE" -eq 1 ]]; then
	GPT_ARGS+=" \
    --num-experts ${NUM_EXPERTS} \
    --expert-model-parallel-size ${EP} \
    --moe-router-topk ${MOE_ROUTER_TOPK} \
    --moe-router-load-balancing-type ${MOE_ROUTER_LOAD_BALANCING_TYPE} \
    --moe-aux-loss-coeff ${MOE_AUX_LOSS_COEFF} \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall \
	--moe-layer-recompute"
fi

export WANDB_MODE=offline
export WANDB_DIR=/logs/wandb

export LOGGING_ARGS="\
    --tensorboard-dir /logs \
    --wandb-project=open_euro_llm \
    --wandb-exp-name=open_euro_llm_${MODEL_SIZE} \
    --wandb-save-dir /logs/wandb"

export CKPT_LOAD_ARGS=""  # Customize if needed
# export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
# export SLURM_CPU_BIND=none

# Generate profile ranks string: "0 1 2 3 4 5 6 7 ... WORLD_SIZE-1"
PROFILE_RANKS=$(seq -s ' ' 0 $((WORLD_SIZE - 1)))
PROFILE_STEPS_OFFSET=3
PREFILE_STEPS=5
PROFILE_START=$((TOTAL_ITERS - PREFILE_STEPS - PROFILE_STEPS_OFFSET))
export PROFILING_ARGS="\
    --profile \
    --profile-step-start $PROFILE_START \
    --profile-step-end $((PROFILE_START + PREFILE_STEPS)) \
    --profile-ranks $PROFILE_RANKS"

# === Launch ===
pretrain_script="/opt/megatron-lm/pretrain_gpt.py"
train_command="torchrun $DISTRIBUTED_ARGS \
    $pretrain_script \
        $GPT_ARGS \
        $DATA_ARGS \
        $EXTRA_ARGS \
        $TRAIN_ARGS \
        $OUTPUT_ARGS \
        $LOGGING_ARGS \
        $CKPT_LOAD_ARGS \
        $PROFILING_ARGS"


# Wrap with Singularity - use train_command_with_nsys which includes nsys
singularity_prefix="singularity exec --nv \
   	--bind /leonardo \
    --bind /leonardo_work \
    --bind "$ABSOLUTE_EXERCISE_DIR":"$ABSOLUTE_EXERCISE_DIR" \
"
gpu_monitor_command="$singularity_prefix \
	$PATH_SINGULARITY python -m utils.gpus_monitor"

#    -B $PATH_RESULTS:/results \ # Bind mount results for saving checkpoints/logs directly from the container
#    -B $PATH_LOGS:/logs \ # Bind mount logs for TensorBoard/WandB logging
train_command="$singularity_prefix \
    -B $PATH_TOKENIZER:/tokenizer \
    -B $PATH_MODEL:/model \
    -B $PATH_DATA:/data \
    -B $PATH_CACHE:/cache \
    -B $PATH_RESULTS:/results \
    -B $PATH_LOGS:/logs \
    -B $EXERCISE_DIR:/workspace/megatron-lm \
    $PATH_SINGULARITY $train_command"

echo "Running command:"
echo "$train_command"

# Create base directory for nsys reports
MODEL_NAME=$(basename "$PATH_MODEL")
export PROFILER_PREFIX_PATH="$EXERCISE_DIR/profiler/\
$MODEL_NAME-$SLURM_JOB_ID-\
n$SLURM_NNODES-\
g4-\
mbs$MBS-\
gbs$GBS-\
tp$TP-\
pp$PP"
if [ -n "$EP" ]; then
    PROFILER_PREFIX_PATH+="-ep$EP"
fi

export GPUS_MONITOR_PREFIX_PATH="$PROFILER_PREFIX_PATH"
if [ $NO_PROFILE -eq 1 ]; then
    echo "$ECHO_PREFIX Profiling is disabled. NSYS output will not be generated."
    NSYS_OUTPUT_DIR="$PROFILER_PREFIX_PATH/no-nsys"
fi
NSYS_OUTPUT_DIR="$PROFILER_PREFIX_PATH/nsys"
if [ -n "$NSYS_OUTPUT_DIR" ]; then
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

    # Wrap training command with nsys - output path will be set per node/rank
    echo "NSYS profiling enabled. Reports will be saved to $NSYS_OUTPUT_DIR"
    train_command="nsys profile \
        -t cuda,osrt,nvtx,cudnn,ucx,cublas \
        --force-overwrite true \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --stats=true \
        --output=${NSYS_OUTPUT_DIR}/profile_node%q{SLURM_NODEID} \
        $train_command"
else
    echo "NSYS_OUTPUT_DIR not set. Profiling disabled."
fi


echo "Executing training with Singularity and nsys profiling..."
echo "$train_command"

train_command="
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
srun -l --ntasks-per-node="$SLURM_NTASKS_PER_NODE" \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    --gres=gpu:4 \
    --export=ALL,NUM_NODES="$SLURM_NNODES" \
    bash -c "$train_command"


if [ $NSYS2PRV -eq 1 ]; then
    experiment_path="$PROFILER_PREFIX_PATH"
    experiment_name=$(basename "$experiment_path")
    echo "$ECHO_PREFIX Generating Paraver traces from NSYS profiles with name $experiment_name..."
    bash "../nsys2prv/nsys2prv-folder.sh" "$NSYS_OUTPUT_DIR" -n "$experiment_name"
fi