#!/bin/bash

# ============================================================
# SLURM Job Submission Wrapper Script
# ============================================================

# Ensure script is run from excercises/ directory
if [[ ! "$(basename "$PWD")" == "excercises" ]]; then
    echo "Error: This script must be run from the excercises/ directory"
    echo "Current directory: $PWD"
    exit 1
fi

usage() {
    echo "Usage: $0 [OPTIONS] [-- EXTRA_ARGS]"
    echo ""
    echo "Options:"
    echo "  -n, --nodes           Number of nodes (default: 1)"
    echo "  -g, --gpus-per-node   GPUs per node (default: 4)"
    echo "  -q, --queue           Queue/QOS for SLURM job (default: acc_bench)"
    echo "  -a, --account         Account name (default: bsc99)"
    echo "  -p, --partition       Partition on HPC (default: acc)"
    echo "  -e, --exercise        Exercise number (1, 2, or 3) (required)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Extra arguments after '--' will be passed to the training script."
    echo ""
    echo "  --slow-dataloading                  Enable slow dataloading example for exercise 1 (DDP) (default: disabled)"
    echo "  --mixed-precision                   Enable mixed precision training using 'bf16' (default: disabled)"
    echo "  --micro-batch-size N                Set micro batch size to N (default: 4)"
    echo "  --gradient-accumulation-steps N     Set gradient accumulation steps to N (default: 1)"
    echo "  --activation-checkpointing          Enable activation checkpointing for exercise 2 (default: disabled)"
    echo ""
    echo "Example:"
    echo "  $0 -n 1 -g 4 -e 1 -a bsc99 -q acc_bench -p acc -- --slow-dataloading --mixed-precision"
    echo ""
    exit 1
}

# Default values
NUM_NODES=1
NUM_GPUS=4
QUEUE="acc_bench"
ACCOUNT="bsc99"
PARTITION="acc"
EXERCISE=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        -g|--gpus-per-node)
            NUM_GPUS="$2"
            shift 2
            ;;
        -q|--queue)
            QUEUE="$2"
            shift 2
            ;;
        -a|--account)
            ACCOUNT="$2"
            shift 2
            ;;
        -p|--partition)
            PARTITION="$2"
            shift 2
            ;;
        -e|--exercise)
            EXERCISE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Error: Unknown option $1"
            usage
            ;;
    esac
done

# All remaining arguments after '--' are extra arguments
EXTRA_ARGS=("$@")

# Validate exercise number
if [[ -z "$EXERCISE" ]]; then
    echo "Error: Exercise number is required (-e, --exercise)"
    usage
fi

if [[ ! "$EXERCISE" =~ ^[0-3]$ ]]; then
    echo "Error: Exercise number must be 1, 2, or 3"
    exit 1
fi

# Set JOB_SCRIPT based on exercise number
case $EXERCISE in
    1)
        JOB_SCRIPT="slurm_nsys.sh"
        EXERCISE_NAME="DDP"
        EXCERCISE_DIR="./excercise_1_DDP"
        ;;
    2)
        JOB_SCRIPT="slurm_nsys.sh"
        EXERCISE_NAME="DeepSpeed"
        EXCERCISE_DIR="./excercise_2_DeepSpeed"
        ;;
    3)
        JOB_SCRIPT="slurm_nsys.sh"
        EXERCISE_NAME="MegatronLM"
        EXCERCISE_DIR="./excercise_3_MegatronLM"
        ;;
esac

i=0
train_config_message="\nTraining configuration added:\n"
messages_to_add=""
while [[ $i -lt ${#EXTRA_ARGS[@]} ]]; do
    arg="${EXTRA_ARGS[$i]}"
    case "$arg" in
        --slow-dataloading)
            export SLOW_DATALOADING=1
            messages_to_add+="  * Slow dataloading mode enabled!\n"
            ;;
        --mixed-precision)
            export MIXED_PRECISION=1
            messages_to_add+="  * Mixed precision mode enabled!\n"
            ;;
        --micro-batch-size)
            ((i++))
            export MICRO_BATCH_SIZE="${EXTRA_ARGS[$i]}"
            messages_to_add+="  * Micro batch size set to $MICRO_BATCH_SIZE.\n"
            ;;
        --gradient-accumulation-steps)
            ((i++))
            export GRADIENT_ACCUMULATION_STEPS="${EXTRA_ARGS[$i]}"
            messages_to_add+="  * Gradient accumulation steps set to $GRADIENT_ACCUMULATION_STEPS.\n"
            ;;
        --activation-checkpointing)
            export ACTIVATION_CHECKPOINTING=1 
            messages_to_add+="  * Activation checkpointing enabled!\n" ;;
        *)
            echo "  !! Error: Unknown option $arg !!"
            usage
            ;;
    esac
    ((i++))
done

if [[ -z "$messages_to_add" ]]; then
    messages_to_add="$train_config_message  * Default training configuration will be used.\n    No extra training specific arguments were provided.\n"
else
    messages_to_add="$train_config_message $messages_to_add"
fi

# Slow dataloading is only valid for exercise 1 (DDP)
if [[ $SLOW_DATALOADING -eq 1 && $EXERCISE -ne 1 ]]; then
    echo "Error: --slow-dataloading option is only valid for exercise 1"
    usage
fi

# Check if job script exists
if [[ ! -f "$EXCERCISE_DIR/$JOB_SCRIPT" ]]; then
    echo "Error: Job script not found: $EXCERCISE_DIR/$JOB_SCRIPT"
    exit 1
fi


RANDOM_PREFIX=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 13; echo)
tmp_job_script="$EXCERCISE_DIR/$RANDOM_PREFIX-$JOB_SCRIPT"
cp "$EXCERCISE_DIR/$JOB_SCRIPT" "$tmp_job_script"

# Ensure tmp_config is always deleted on exit (success, failure, or signal)
cleanup() {
    if [[ -f "$tmp_job_script" ]]; then
        rm -f "$tmp_job_script"
        echo "---------------------------------------------------------"
        echo "Cleaned up temporary slurm launch script: $tmp_job_script"
        echo "---------------------------------------------------------"
    fi
}
trap cleanup EXIT
# Export variables for use in SLURM script
export NUM_NODES
export NUM_GPUS
export QUEUE
export ACCOUNT
export PARTITION
export EXERCISE_NAME
export EXCERCISE_DIR
export RANDOM_PREFIX

# Apply placeholders substitution
sed -i "s/nodes={{NUM_NODES}}/nodes=$NUM_NODES/g"  "$tmp_job_script"
sed -i "s/gres=gpu:{{NUM_GPUS}}/gres=gpu:$NUM_GPUS/g"  "$tmp_job_script"
sed -i "s/account={{ACCOUNT}}/account=$ACCOUNT/g"  "$tmp_job_script"
sed -i "s/qos={{QUEUE}}/qos=$QUEUE/g"  "$tmp_job_script"
sed -i "s/partition={{PARTITION}}/partition=$PARTITION/g"  "$tmp_job_script"
sed -i "s|output={{LOG_OUT}}|output=$EXCERCISE_DIR/logs/nodes-$NUM_NODES/%j/log.out|g"  "$tmp_job_script"
sed -i "s|error={{LOG_ERR}}|error=$EXCERCISE_DIR/logs/nodes-$NUM_NODES/%j/log.err|g"  "$tmp_job_script"

echo "============================================================"
echo "NSYS Profiling Configuration (Exercise $EXERCISE - $EXERCISE_NAME):"
echo "  Number of Nodes: $NUM_NODES"
echo "  GPUs per Node: $NUM_GPUS"
echo "  Total GPUs: $((NUM_NODES * NUM_GPUS))"
echo "  Account: $ACCOUNT"
echo "  Queue/QOS: $QUEUE"
echo "  Partition: $PARTITION"
echo "  Job Script: $tmp_job_script"
echo -e "$messages_to_add"
echo "============================================================"

JOB_ID=$(sbatch --export=ALL "$tmp_job_script" | awk '{print $NF}')

echo "Submitted job with ID=$JOB_ID"
echo ""
echo "Monitor with: squeue -j $JOB_ID"
echo "Logs will be at: $EXCERCISE_DIR/logs/nodes-$NUM_NODES/$JOB_ID/"
echo "Profiles will be at: $EXCERCISE_DIR/profiler/$JOB_ID-nsys/"

# Restore placeholders
# EXCERCISE_DIR_ESCAPED=$(echo "$EXCERCISE_DIR" | sed 's/\./\\./g')
# sed -i "s|output=$EXCERCISE_DIR_ESCAPED/logs/nodes-$NUM_NODES/%j/log\.out|output={{LOG_OUT}}|g"  "$JOB_SCRIPT"
# sed -i "s|error=$EXCERCISE_DIR_ESCAPED/logs/nodes-$NUM_NODES/%j/log\.err|error={{LOG_ERR}}|g"  "$JOB_SCRIPT"
# sed -i "s/nodes=$NUM_NODES/nodes={{NUM_NODES}}/g"  "$JOB_SCRIPT"
# sed -i "s/gres=gpu:$NUM_GPUS/gres=gpu:{{NUM_GPUS}}/g"  "$JOB_SCRIPT"
# sed -i "s/account=$ACCOUNT/account={{ACCOUNT}}/g"  "$JOB_SCRIPT"
# sed -i "s/qos=$QUEUE/qos={{QUEUE}}/g"  "$JOB_SCRIPT"
# sed -i "s/partition=$PARTITION/partition={{PARTITION}}/g"  "$JOB_SCRIPT"