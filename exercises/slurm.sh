#!/bin/bash

# ============================================================
# SLURM Job Submission Wrapper Script
# ============================================================

# Ensure script is run from exercises/ directory
if [[ ! "$(basename "$PWD")" == "exercises" ]]; then
    echo "Error: This script must be run from the 'exercises/' directory"
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
    echo "  -e, --exercise        Exercise number (0, 1, 2, or 3) (required)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Extra arguments after '--' will be passed to the training script."
    echo ""
    echo "  --slow-dataloading                  Enable slow dataloading example for exercise 1 (DDP) (default: disabled)"
    echo "  --mixed-precision                   Enable mixed precision training using 'bf16' (default: disabled)"
    echo "  --micro-batch-size N                Set micro batch size to N (default: 4)"
    echo "  --gradient-accumulation-steps N     Set gradient accumulation steps to N (default: 1)"
    echo "  --activation-checkpointing          Enable activation/gradient checkpointing (default: disabled)"
    echo "  --ds-hpz-partition N                Set DeepSpeed HPZ partition size to N, for exercise 2 (DeepSpeed) only. Refers to num of GPUs per model replica (default: 4)"
    echo "  --ds-heavy-comm                     Enable example of heavier communication chuck sizes in DeepSpeed, for exercise 2 only (default: disabled)"
    echo "  --ds-stage2                         Use DeepSpeed stage 2 partitioning instead of stage 3, for exercise 2 (DeepSpeed) only (default: stage 3)"
    echo "  --ds-no-overlap                     Disable overlap communication in DeepSpeed, for exercise 2 only (default: disabled)"
    echo "  --tp N                              Set tensor parallelism to N, for exercise 3 (MegatronLM) only (default: 1, i.e. no tensor parallelism)"
    echo "  --pp N                              Set pipeline parallelism to N, for exercise 3 (MegatronLM) only (default: 1, i.e. no pipeline parallelism)"
    echo "  --global-batch-size N               Set global batch size to N, for exercise 3 (MegatronLM) only (default: 16)"
    echo "  --no-profile                        Disable profiling with NSYS (default: profiling enabled)"
    echo "  --nsys2prv                          After profiling, automatically translate NSYS output to Paraver traces using nsys2prv (default: disabled)"   
    echo ""
    echo "Example:"
    echo "  $0 -n 1 -g 4 -e 1 -a tra26_minwinsc -p boost_usr_prod -- --mixed-precision --gradient-accumulation-steps 4"
    echo ""
    exit 1
}

# Default values
NUM_NODES=1
NUM_GPUS=4
QUEUE="qos_prio"
ACCOUNT="tra26_minwinsc"
PARTITION="boost_usr_prod"
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
    echo "Error: Exercise number must be 0, 1, 2, or 3"
    exit 1
fi

# Set JOB_SCRIPT based on exercise number
case $EXERCISE in
    0) 
        JOB_SCRIPT="tests.sh"
        EXERCISE_NAME="Communication Tests"
        EXERCISE_DIR="./exercise_0_Communication_Tests"
        ;;
    1)
        JOB_SCRIPT="slurm_nsys.sh"
        EXERCISE_NAME="DDP"
        EXERCISE_DIR="./exercise_1_DDP"
        ;;
    2)
        JOB_SCRIPT="slurm_nsys.sh"
        EXERCISE_NAME="DeepSpeed"
        EXERCISE_DIR="./exercise_2_DeepSpeed"
        ;;
    3)
        JOB_SCRIPT="slurm_nsys.sh"
        EXERCISE_NAME="MegatronLM"
        EXERCISE_DIR="./exercise_3_MegatronLM"
        ;;
esac

i=0

# if exercise 0, no extra arguments should be provided
if [[ $EXERCISE -eq 0 && ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "Error: Exercise 0 does not accept extra arguments. Please remove the following extra arguments: ${EXTRA_ARGS[*]}"
    usage
fi

train_config_message="\nTraining configuration added:\n"
messages_to_add=""

#DS_NO_OVERLAP=0
#DS_STAGE2=0
#HEAVY_COMM=0
#ACTIVATION_CHECKPOINTING=0
#MIXED_PRECISION=0
#SLOW_DATALOADING=0
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
        --ds-hpz-partition)
            ((i++))
            export HPZ_PARTITION_SIZE="${EXTRA_ARGS[$i]}"
            messages_to_add+="  * DeepSpeed HPZ partition size set to $HPZ_PARTITION_SIZE.\n"
            ;;
        --ds-heavy-comm)
            export HEAVY_COMM=1
            messages_to_add+="  * DeepSpeed example of heavy communication overhead enabled!\n"
            ;;
        --ds-stage2)
            export DS_STAGE2=1
            messages_to_add+="  * DeepSpeed using stage 2 partitioning!\n"
            ;;
        --no-profile)
            export NO_PROFILE=1
            messages_to_add+="  * Profiling disabled!\n"
            ;;
        --ds-no-overlap)
            export DS_NO_OVERLAP=1
            messages_to_add+="  * DeepSpeed overlap communication disabled!\n"
            ;;
        --tp)
            ((i++))
            export TP="${EXTRA_ARGS[$i]}"
            messages_to_add+="  * Tensor parallelism set to $TP!\n"
            ;;
        --pp)
            ((i++))
            export PP="${EXTRA_ARGS[$i]}"
            messages_to_add+="  * Pipeline parallelism set to $PP!\n"
            ;;
        --global-batch-size)
            ((i++))
            export GLOBAL_BATCH_SIZE="${EXTRA_ARGS[$i]}"
            messages_to_add+="  * Global batch size set to $GLOBAL_BATCH_SIZE!\n"
            ;;
        --nsys2prv)
            export NSYS2PRV=1
            messages_to_add+="  * nsys2prv translation enabled!\n"
            ;;
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
    messages_to_add="$train_config_message$messages_to_add"
fi

STAGE=3 # default to stage 3 for DeepSpeed, can be overridden with --ds-stage2
if [[ -n "$DS_STAGE2" ]]; then
    STAGE=2
fi
# Set JOB_SCRIPT based on exercise number
#case $EXERCISE in
#    1)  
#    MODEL_NAME="Llama-3.1-1B"
#    PROFILE_DIR="$MODEL_NAME-$SLURM_JOB_ID-\
#n$NUM_NODES-\
#g4-\
#mbs$MICRO_BATCH_SIZE-\
#gas${GRADIENT_ACCUMULATION_STEPS}-\
#mixed${MIXED_PRECISION}-\
#slow${SLOW_DATALOADING}/nsys"
#        ;;
#    2) 
#    MODEL_NAME="Mistral-7B-v0.1"
#    PROFILE_DIR="$MODEL_NAME-$SLURM_JOB_ID-\
#n$NUM_NODES-\
#g4-\
#mbs$MICRO_BATCH_SIZE-\
#gas${GRADIENT_ACCUMULATION_STEPS}-\
#mixed${MIXED_PRECISION}-\
#actckpt${ACTIVATION_CHECKPOINTING}-\
#commoverlap${DS_NO_OVERLAP}-\
#hpz${HPZ_PARTITION_SIZE}-\
#ZeRO${STAGE}/nsys"
#        ;;
#    3) 
#    MODEL_NAME="Mistral-7B-v0.1"
#    PROFILE_DIR="$MODEL_NAME-$SLURM_JOB_ID-\
#n$NUM_NODES-\
#g4-\
#mbs$MICRO_BATCH_SIZE-\
#gbs$GLOBAL_BATCH_SIZE-\
#tp$TP-\
#pp$PP/nsys"
#        ;;
#esac


# Slow dataloading is only valid for exercise 1 (DDP)
if [[ $SLOW_DATALOADING -eq 1 && $EXERCISE -ne 1 ]]; then
    echo "Error: --slow-dataloading option is only valid for exercise 1"
    echo ""
    usage
fi

if [[ $EXERCISE -eq 1 && -n "$ACTIVATION_CHECKPOINTING" ]]; then
    echo "Error: --activation-checkpointing is not a valid option for exercise $EXERCISE !!"
    echo ""
    usage
fi

if [[ -n "$HPZ_PARTITION_SIZE" && $EXERCISE -ne 2 ]]; then
    echo "Error: --ds-hpz-partition option is only valid for exercise 2 (DeepSpeed)"
    echo ""
    usage
fi

if [[ -n "$DS_STAGE2" && $EXERCISE -ne 2 ]]; then
    echo "Error: --ds-stage2-partition option is only valid for exercise 2 (DeepSpeed)"
    echo ""
    usage
fi

if [[ -n "$HEAVY_COMM" && $EXERCISE -ne 2 ]]; then
    echo "Error: --ds-heavy-comm option is only valid for exercise 2 (DeepSpeed)"
    echo ""
    usage
fi
if [[ -n "$DS_NO_OVERLAP" && $EXERCISE -ne 2 ]]; then
    echo "Error: --ds-no-overlap option is only valid for exercise 2 (DeepSpeed)"
    echo ""
    usage
fi
if [[ $ACTIVATION_CHECKPOINTING -eq 1 && $MIXED_PRECISION -eq 0 ]]; then
    echo "Error: Activation checkpointing with full precision is not supported in the provided configs.Please enable mixed precision or disable activation checkpointing."
    echo ""
    usage
fi
if [[ $HEAVY_COMM -eq 1 && $MIXED_PRECISION -eq 0 ]]; then
    echo "Error: Heavy communication overhead example is only supported with mixed precision in the provided configs. Please enable mixed precision to use this option."
    echo ""
    usage
fi
if [[ $HEAVY_COMM -eq 1 && $ACTIVATION_CHECKPOINTING -eq 1 ]]; then
    echo "Error: Heavy communication overhead example is not compatible with activation checkpointing in the provided configs. Please disable activation checkpointing to use this option."
    echo ""
    usage
fi
# Check if job script exists
if [[ ! -f "$EXERCISE_DIR/$JOB_SCRIPT" ]]; then
    echo "Error: Job script not found: $EXERCISE_DIR/$JOB_SCRIPT"
    echo "Contact the workshop organizers to resolve this issue."
    echo ""
    exit 1
fi

if [[ -n $PP ]]; then
    if [[ $EXERCISE -ne 3 ]]; then
        echo "Error: --pp option is only valid for exercise 3 (MegatronLM)"
        echo ""
        usage
    fi
    if [[ $PP -gt $NUM_NODES ]]; then
        echo "Error: Invalid value for --pp option. PP($PP) cannot be greater than the number of nodes ($NUM_NODES)."
        echo ""
        usage
    fi
fi

if [[ -n $GLOBAL_BATCH_SIZE && $EXERCISE -ne 3 ]]; then
    echo "Error: --global-batch-size option is only valid for exercise 3 (MegatronLM)"
    echo ""
    usage
fi

if [[ -n $TP && $EXERCISE -ne 3 ]]; then
    echo "Error: --tp option is only valid for exercise 3 (MegatronLM)"
    echo ""
    usage
fi



RANDOM_PREFIX=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 13; echo)
tmp_job_script="$EXERCISE_DIR/$RANDOM_PREFIX-$JOB_SCRIPT"
cp "$EXERCISE_DIR/$JOB_SCRIPT" "$tmp_job_script"

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
export EXERCISE_DIR
export RANDOM_PREFIX

# Apply placeholders substitution
sed -i "s/nodes={{NUM_NODES}}/nodes=$NUM_NODES/g"  "$tmp_job_script"
sed -i "s/gres=gpu:{{NUM_GPUS}}/gres=gpu:$NUM_GPUS/g"  "$tmp_job_script"
sed -i "s/account={{ACCOUNT}}/account=$ACCOUNT/g"  "$tmp_job_script"
sed -i "s/qos={{QUEUE}}/qos=$QUEUE/g"  "$tmp_job_script"
sed -i "s/partition={{PARTITION}}/partition=$PARTITION/g"  "$tmp_job_script"
sed -i "s|output={{LOG_OUT}}|output=$EXERCISE_DIR/logs/nodes-$NUM_NODES/%j/log.out|g"  "$tmp_job_script"
sed -i "s|error={{LOG_ERR}}|error=$EXERCISE_DIR/logs/nodes-$NUM_NODES/%j/log.err|g"  "$tmp_job_script"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
MAGENTA='\033[1;35m'
CYAN='\033[1;36m'
BOLD='\033[1m'
RESET='\033[0m'


echo -e "${BOLD}${CYAN}============================================================${RESET}"
echo -e "${BOLD}${GREEN}NSYS Profiling Configuration${RESET} (${YELLOW}Exercise $EXERCISE - $EXERCISE_NAME${RESET}):"
echo -e "  ${MAGENTA}Number of Nodes:${RESET} ${BOLD}$NUM_NODES${RESET}"
echo -e "  ${MAGENTA}GPUs per Node:${RESET} ${BOLD}$NUM_GPUS${RESET}"
echo -e "  ${MAGENTA}Total GPUs:${RESET} ${BOLD}$((NUM_NODES * NUM_GPUS))${RESET}"
echo -e "  ${MAGENTA}Account:${RESET} ${BOLD}$ACCOUNT${RESET}"
echo -e "  ${MAGENTA}Queue/QOS:${RESET} ${BOLD}$QUEUE${RESET}"
echo -e "  ${MAGENTA}Partition:${RESET} ${BOLD}$PARTITION${RESET}"
echo -e "  ${MAGENTA}Job Script:${RESET} ${BOLD}$tmp_job_script${RESET}"
echo -e "${BLUE}$messages_to_add${RESET}"
echo -e "${BOLD}${CYAN}============================================================${RESET}"

if [[ $EXERCISE -eq 0 ]]; then
    echo -e "${BOLD}${YELLOW}Running communication tests for intra-node...${RESET}"
    srun --nodes=$NUM_NODES \
        --ntasks-per-node=1 \
        --cpus-per-task=32 --gres=gpu:4 \
        --account=$ACCOUNT --partition=$PARTITION \
        "$EXERCISE_DIR/$JOB_SCRIPT"
    echo "Communication tests completed!"
    exit 0
fi
JOB_ID=$(sbatch --export=ALL "$tmp_job_script" | awk '{print $NF}')

echo -e "${BOLD}${GREEN}Submitted job with ID=${RESET}${YELLOW}$JOB_ID${RESET}"
echo ""
echo -e "${CYAN}Monitor with:${RESET} ${BOLD}squeue -j $JOB_ID${RESET}"
echo -e "${CYAN}Logs will be at:${RESET} ${BOLD}$EXERCISE_DIR/logs/nodes-$NUM_NODES/$JOB_ID/${RESET}"
echo -e "${CYAN}Profiles will be at:${RESET} ${BOLD}$EXERCISE_DIR/profiler/<model name>-$JOB_ID-<training configuration>${RESET}"
