#!/bin/bash

# ============================================================
# SLURM Job Submission Wrapper Script
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
MAGENTA='\033[1;35m'
CYAN='\033[1;36m'
BOLD='\033[1m'
RESET='\033[0m'

# Ensure script is run from exercises/ directory
if [[ ! "$(basename "$PWD")" == "exercises" ]]; then
    echo "---------------------------------------------------------"
    echo -e "${BOLD}${RED}Error:${RESET} ${RED}This script must be run from the 'exercises/' directory${RESET}"
    echo -e "${BOLD}${YELLOW}Current directory:${RESET} $PWD"
    echo "---------------------------------------------------------"
    exit 1
fi

error_usage() {
    echo "---------------------------------------------------------"
    echo -e "${BOLD}${RED}Error:${RESET} $1"
    echo ""
    echo -e "> Run ${YELLOW}'bash $0 --help'${RESET} for usage instructions."
    echo "---------------------------------------------------------"
    exit 1
}

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
    echo "  --model NAME                        Set model name to profile  "
    echo "  --slow-dataloading                  Enable slow dataloading example for exercise 1 (DDP) (default: disabled)"
    echo "  --mixed-precision TYPE              Enable mixed precision training using 'bf16', 'fp16' or 'fp8' (default: disabled)"
    echo "  --micro-batch-size N                Set micro batch size to N (default: 4)"
    echo "  --gradient-accumulation-steps N     Set gradient accumulation steps to N (default: 1)"
    echo "  --activation-checkpointing          Enable activation/gradient checkpointing (default: disabled)"
    echo "  --ds-hpz-partition N                Set DeepSpeed HPZ partition size to N, for exercise 2 (DeepSpeed) only. Refers to num of GPUs per model replica (default: 4)"
    echo "  --ds-heavy-comm                     Enable example of heavier communication chuck sizes in DeepSpeed, for exercise 2 only (default: disabled)"
    echo "  --ds-stage2                         Use DeepSpeed stage 2 partitioning instead of stage 3, for exercise 2 (DeepSpeed) only (default: stage 3)"
    echo "  --ds-no-overlap                     Disable overlap communication in DeepSpeed, for exercise 2 only (default: disabled)"
    echo "  --tp N                              Set tensor parallelism to N, for exercise 3 (MegatronLM) only (default: 1, i.e. no tensor parallelism)"
    echo "  --pp N                              Set pipeline parallelism to N, for exercise 3 (MegatronLM) only (default: 1, i.e. no pipeline parallelism)"
    echo "  --cp N                              Set context parallelism to N, for exercise 3 (MegatronLM) only (default: 1, i.e. no context parallelism)"
    echo "  --ep N                              Set expert parallelism to N, for exercise 3 (MegatronLM) only (default: 1, i.e. no expert parallelism)"
    echo "  --global-batch-size N               Set global batch size to N, for exercise 3 (MegatronLM) only (default: 16)"
    echo "  --no-profile                        Disable profiling with NSYS (default: profiling enabled)"
    echo "  --nsys2prv                          After profiling, automatically translate NSYS output to Paraver traces using nsys2prv (default: disabled)"   
    echo "  --models_list                       List available models for profiling for given exercise."
    echo ""
    echo "Example:"
    echo "  $0 -n 1 -g 4 -e 1 -a bsc99 -q acc_bench -p acc -- --model Llama_32_3B --mixed-precision bf16 --micro-batch-size 4 --gradient-accumulation-steps 4"
    echo ""
    exit 1
}

source env.sh

# Default values
NUM_NODES=1
NUM_GPUS=4
QUEUE="$QUEUE"
ACCOUNT="$ACCOUNT"
PARTITION="$PARTITION"
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
            message="Unknown option $1"
            error_usage "$message"
            ;;
    esac
done

# All remaining arguments after '--' are extra arguments
EXTRA_ARGS=("$@")

# Validate exercise number
if [[ -z "$EXERCISE" ]]; then
    message="Exercise number is required (-e, --exercise)"
    error_usage "$message"
fi

if [[ ! "$EXERCISE" =~ ^[0-3]$ ]]; then
    message="Exercise number must be 0, 1, 2, or 3"
    error_usage "$message"
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

ABSOLUTE_EXERCISE_DIR="$(realpath "$EXERCISE_DIR")"

i=0

# if exercise 0, no extra arguments should be provided
if [[ $EXERCISE -eq 0 && ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    message="Exercise 0 does not accept extra arguments. Please remove the following extra arguments: ${EXTRA_ARGS[*]}"
    error_usage "$message"
fi

train_config_message="\nTraining configuration added:\n"
messages_to_add=""

#DS_NO_OVERLAP=0
#DS_STAGE2=0
#HEAVY_COMM=0
#ACTIVATION_CHECKPOINTING=0
#MIXED_PRECISION=0
#SLOW_DATALOADING=0


declare -A EX1_MODELS
declare -A EX2_MODELS
declare -A EX3_MODELS
declare -A EX3_GPT_ARGS

source models.sh



while [[ $i -lt ${#EXTRA_ARGS[@]} ]]; do
    arg="${EXTRA_ARGS[$i]}"
    case "$arg" in
        --model)
            ((i++))
            export MODEL_NAME="${EXTRA_ARGS[$i]}"
            messages_to_add+="  * Model set to $MODEL_NAME.\n"
            ;;
        --slow-dataloading)
            export SLOW_DATALOADING=1
            messages_to_add+="  * Slow dataloading mode enabled!\n"
            ;;
        --mixed-precision)
            export MIXED_PRECISION=1
            ((i++))
            PRECISION_TYPE="${EXTRA_ARGS[$i]}"
            # Validate precision type
            if [[ ! "$PRECISION_TYPE" =~ ^(fp16|bf16|fp8)$ ]]; then
                echo "Error: Invalid mixed precision type '$PRECISION_TYPE'"
                echo "Valid options: fp16, bf16, fp8"
                exit 1
            fi
            export PRECISION_TYPE="$PRECISION_TYPE"
            messages_to_add+="  * Mixed precision mode enabled using '$PRECISION_TYPE'!\n"
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
            #export ACTIVATION_CHECKPOINTING=1 
            export RECOMPUTE=1
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
        --cp)
            ((i++))
            export CP="${EXTRA_ARGS[$i]}"
            messages_to_add+="  * Context parallelism set to $CP!\n"
            ;;
        --ep)
            ((i++))
            export EP="${EXTRA_ARGS[$i]}"
            messages_to_add+="  * Expert parallelism set to $EP!\n"
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
        --models_list)
            if [[ $EXERCISE -eq 1 ]]; then
                echo "Available models for Exercise 1 (DDP):"
                for model in "${!EX1_MODELS[@]}"; do
                    echo "  - $model"
                done
            elif [[ $EXERCISE -eq 2 ]]; then
                echo "Available models for Exercise 2 (DeepSpeed):"
                for model in "${!EX2_MODELS[@]}"; do
                    echo "  - $model"
                done
            elif [[ $EXERCISE -eq 3 ]]; then
                echo "Available models for Exercise 3 (MegatronLM):"
                for model in "${!EX3_MODELS[@]}"; do
                    echo "  - $model: ${EX3_MODELS[$model]}"
                done
            else
                echo "No models available for Exercise $EXERCISE."
            fi
            exit 0
            ;;
        *)
            message="Unknown option '$arg' !!"
            error_usage "$message"
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


# Validate model exists in the appropriate exercise model list
if [[ -n "$MODEL_NAME" ]]; then
    case $EXERCISE in
        1)
            if [[ ! -v EX1_MODELS["$MODEL_NAME"] ]]; then
                echo "Error: Model '$MODEL_NAME' is not available for Exercise 1 (DDP)"
                echo "Available models:"
                for model in "${!EX1_MODELS[@]}"; do
                    echo "  - $model"
                done
                exit 1
            fi
            export PATH_MODEL="${EX1_MODELS[$MODEL_NAME]}"
            export PATH_TOKENIZER="${EX1_MODELS[$MODEL_NAME]}"
            ;;
        2)
            if [[ ! -v EX2_MODELS["$MODEL_NAME"] ]]; then
                echo "Error: Model '$MODEL_NAME' is not available for Exercise 2 (DeepSpeed)"
                echo "Available models:"
                for model in "${!EX2_MODELS[@]}"; do
                    echo "  - $model"
                done
                exit 1
            fi
            export PATH_MODEL="${EX2_MODELS[$MODEL_NAME]}"
            export PATH_TOKENIZER="${EX2_MODELS[$MODEL_NAME]}"
            ;;
        3)
            if [[ ! -v EX3_MODELS["$MODEL_NAME"] ]]; then
                echo "Error: Model '$MODEL_NAME' is not available for Exercise 3 (MegatronLM)"
                echo "Available models:"
                for model in "${!EX3_MODELS[@]}"; do
                    echo "  - $model"
                done
                exit 1
            fi
            # Also validate GPT args file exists for exercise 3
            if [[ ! -v EX3_GPT_ARGS["$MODEL_NAME"] ]]; then
                echo "Error: No GPT args configuration found for model '$MODEL_NAME'"
                exit 1
            fi

            export GPT_ARGS_FILE="$ABSOLUTE_EXERCISE_DIR/${EX3_GPT_ARGS[$MODEL_NAME]}"
            export PATH_MODEL="${EX3_MODELS[$MODEL_NAME]}"
            export PATH_TOKENIZER="${EX3_MODELS[$MODEL_NAME]}"
            #echo "GPT args file set to $GPT_ARGS_FILE based on provided model name ${MODEL_NAME}"
            ;;
    esac
else
    echo "Error: --model option is required to specify which model to profile for exercise $EXERCISE"
    echo "Use --models_list option to see available models for this exercise."
    exit 1
fi


# Slow dataloading is only valid for exercise 1 (DDP)
if [[ $SLOW_DATALOADING -eq 1 && $EXERCISE -ne 1 ]]; then
    message="--slow-dataloading option is only valid for exercise 1"
    error_usage "$message"
fi

if [[ $EXERCISE -eq 1 && -n "$ACTIVATION_CHECKPOINTING" ]]; then
    message="--activation-checkpointing is not a valid option for exercise $EXERCISE !!"
    error_usage "$message"
fi

if [[ -n "$HPZ_PARTITION_SIZE" && $EXERCISE -ne 2 ]]; then
    message="--ds-hpz-partition option is only valid for exercise 2 (DeepSpeed)"
    error_usage "$message"
fi

if [[ -n "$DS_STAGE2" && $EXERCISE -ne 2 ]]; then
    message="--ds-stage2-partition option is only valid for exercise 2 (DeepSpeed)"
    error_usage "$message"
fi

if [[ -n "$HEAVY_COMM" && $EXERCISE -ne 2 ]]; then
    message="--ds-heavy-comm option is only valid for exercise 2 (DeepSpeed)"
    error_usage "$message"
fi
if [[ -n "$DS_NO_OVERLAP" && $EXERCISE -ne 2 ]]; then
    message="--ds-no-overlap option is only valid for exercise 2 (DeepSpeed)"
    error_usage "$message"
fi
if [[ $ACTIVATION_CHECKPOINTING -eq 1 && $MIXED_PRECISION -eq 0 ]]; then
    message="Activation checkpointing with full precision is not supported in the provided configs.Please enable mixed precision or disable activation checkpointing."
    error_usage "$message"
fi
if [[ $HEAVY_COMM -eq 1 && $MIXED_PRECISION -eq 0 ]]; then
    message="Heavy communication overhead example is only supported with mixed precision in the provided configs. Please enable mixed precision to use this option."
    error_usage "$message"
fi
if [[ $HEAVY_COMM -eq 1 && $ACTIVATION_CHECKPOINTING -eq 1 ]]; then
    message="Heavy communication overhead example is not compatible with activation checkpointing in the provided configs. Please disable activation checkpointing to use this option."
    error_usage "$message"
fi
# Check if job script exists
if [[ ! -f "$EXERCISE_DIR/$JOB_SCRIPT" ]]; then
    message="Job script not found: '$EXERCISE_DIR/$JOB_SCRIPT'! Contact the workshop organizers to resolve this issue."
    error_usage "$message"
fi

if [[ -n $PP ]]; then
    if [[ $EXERCISE -ne 3 ]]; then
        message="--pp option is only valid for exercise 3 (MegatronLM)"
        error_usage "$message"
    fi
    if [[ $PP -gt $NUM_NODES ]]; then
        message="Invalid value for --pp option. PP($PP) cannot be greater than the number of nodes ($NUM_NODES)."
        error_usage "$message"
    fi
fi

if [[ -n $GLOBAL_BATCH_SIZE && $EXERCISE -ne 3 ]]; then
    message="--global-batch-size option is only valid for exercise 3 (MegatronLM)"
    error_usage "$message"
fi

if [[ -n $TP && $EXERCISE -ne 3 ]]; then
    message="--tp option is only valid for exercise 3 (MegatronLM)"
    error_usage "$message"
fi

if [[ -n $CP && $EXERCISE -ne 3 ]]; then
    message="--cp option is only valid for exercise 3 (MegatronLM)"
    error_usage "$message"
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
export ABSOLUTE_EXERCISE_DIR
export NUM_NODES
export NUM_GPUS
export QUEUE
export ACCOUNT
export PARTITION
export EXERCISE_NAME
export EXERCISE_DIR
export RANDOM_PREFIX
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Apply placeholders substitution
sed -i "s/nodes={{NUM_NODES}}/nodes=$NUM_NODES/g"  "$tmp_job_script"
sed -i "s/gres=gpu:{{NUM_GPUS}}/gres=gpu:$NUM_GPUS/g"  "$tmp_job_script"
sed -i "s/account={{ACCOUNT}}/account=$ACCOUNT/g"  "$tmp_job_script"
sed -i "s/qos={{QUEUE}}/qos=$QUEUE/g"  "$tmp_job_script"
sed -i "s/partition={{PARTITION}}/partition=$PARTITION/g"  "$tmp_job_script"
sed -i "s|output={{LOG_OUT}}|output=$EXERCISE_DIR/logs/$MODEL_NAME/nodes-$NUM_NODES/%j/log.out|g"  "$tmp_job_script"
sed -i "s|error={{LOG_ERR}}|error=$EXERCISE_DIR/logs/$MODEL_NAME/nodes-$NUM_NODES/%j/log.err|g"  "$tmp_job_script"

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
        --cpus-per-task=80 --gres=gpu:4 \
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
echo -e "${CYAN}Profiles will be at:${RESET} ${BOLD}$EXERCISE_DIR/profiler/$MODEL_NAME/$JOB_ID-<training configuration>${RESET}"
