#!/bin/bash
#SBATCH --job-name=nsys2prv_folder
#SBATCH --output=/leonardo/home/userexternal/apaliour/training-profiling-workshop/nsys2prv/slurm-logs/%j/log.out
#SBATCH --error=/leonardo/home/userexternal/apaliour/training-profiling-workshop/nsys2prv/slurm-logs/%j/log.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --account=tra26_minwinsc
#SBATCH --partition=boost_usr_prod

module purge
module load cuda/12.6

CONTAINER="/leonardo/home/userexternal/apaliour/training-profiling-workshop/singularity-images/ai-profiling-workshop-nsys2prv.sif"

set -euo pipefail

usage() {
    YELLOW="\033[1;33m"
    CYAN="\033[1;36m"
    GREEN="\033[1;32m"
    RED="\033[1;31m"
    NC="\033[0m" # No Color

    echo -e "${YELLOW}Usage:${NC} $0 <folder-path> [OPTIONS]"
    echo ""
    echo -e "${CYAN}Arguments:${NC}"
    echo -e "  <folder-path>                Directory containing .nsys-rep files to convert (${RED}REQUIRED, must be first argument${NC})"
    echo -e "${CYAN}Options:${NC}"
    echo -e "  -o, --output-folder          Directory for output .prv files (default: same as input)"
    echo -e "  -n, --name                   Name for output .prv file (default: input folder name)"
    echo -e "  -f, --file-by-file           Convert each .nsys-rep file individually (default: merge all files)"
    echo -e "  -h, --help                   Show this help message"
    echo -e "\n${GREEN}# EXAMPLE 1:${NC} Output to same folder, use folder name as output name"
    echo -e "  sbatch nsys2prv-folder.sh /path/to/nsys_traces"
    echo -e "${GREEN}# EXAMPLE 2:${NC} Specify output folder"
    echo -e "  sbatch nsys2prv-folder.sh /path/to/nsys-folder -o /path/to/output"
    echo -e "${GREEN}# EXAMPLE 3:${NC} Specify output folder and custom output name"
    echo -e "  sbatch nsys2prv-folder.sh /path/to/nsys-folder -o /path/to/output -n my-trace-name\n"
    exit 1
}

# Ensure at least one argument (the folder path) is provided
if [[ $# -lt 1 ]]; then
    echo "Error: <folder-path> argument is required."
    usage
fi

module purge
module load cuda/12.6 # -> nsys version 2024.6.2 compatible with nemo25.02 cuda/nsys version

# Read the first positional argument as the folder path
INPUT_DIR="$1"
shift

# Default values
FILE_BY_FILE=false
OUTPUT_DIR=""
OUTPUT_NAME=""

# Parse the rest as flags
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output-folder)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--name)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        -f|--file-by-file)
            FILE_BY_FILE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown option $1"
            usage
            ;;
    esac
done

SINGULARITYENV_NSYS_HOME=$(which nsys)
SINGULARITYENV_NSYS_HOME=$(echo "$SINGULARITYENV_NSYS_HOME" | sed 's|bin/nsys||')
export SINGULARITYENV_NSYS_HOME
export SINGULARITYENV_APPEND_PATH="$(which nsys)"
#if [[ ! "$(basename "$PWD")" == "nsys2prv" ]]; then
#    echo "Error: This script must be run from the nsys2prv/ directory"
#    echo "Current directory: $PWD"
#    exit 1
#fi


SINGU_PREFIX="singularity exec --nv -B leonard/ $CONTAINER"

$SINGU_PREFIX bash -c "echo \"PATH inside container: \$PATH\"; echo \"NSYS_HOME inside container: \$NSYS_HOME\""


INPUT_DIR="$(realpath "$INPUT_DIR")"

# Validate input directory
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ERROR: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Create output directory if needed (before resolving path)
OUTPUT_DIR="${2:-$INPUT_DIR}"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
mkdir -p "$OUTPUT_DIR"

# Find all .nsys-rep files
NSYS_FILES=( "$INPUT_DIR"/*.nsys-rep )

if [[ ${#NSYS_FILES[@]} -eq 0 || ! -e "${NSYS_FILES[0]}" ]]; then
    echo "ERROR: No .nsys-rep files found in $INPUT_DIR"
    exit 1
fi

# Derive output name from folder name if not provided
OUTPUT_NAME="${3:-$(basename "$INPUT_DIR")}"
OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT_NAME"

# Trace types to extract
TRACE_TYPES="nvtx_pushpop_trace,nccl,cuda_api_trace,gpu_metrics,cuda_gpu_trace"

echo "========================================"
echo "NSYS to Paraver Conversion"
echo "========================================"
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Output name:      $OUTPUT_NAME"
echo "Files to convert: ${#NSYS_FILES[@]}"
echo "Trace types:      $TRACE_TYPES"
echo "Conversion mode:  $([ "$FILE_BY_FILE" = true ] && echo "File-by-file" || echo "Merged")"
echo "========================================"
echo ""
echo "Input files:"
for f in "${NSYS_FILES[@]}"; do
    echo "  - $(basename "$f")"
done
echo ""

cd "$INPUT_DIR"

if [ "$FILE_BY_FILE" = true ]; then
    # Convert each file individually
    echo "Converting files individually..."
    echo ""
    
    SUCCESS_COUNT=0
    FAIL_COUNT=0
    
    for nsys_file in "${NSYS_FILES[@]}"; do
        file_basename=$(basename "$nsys_file" .nsys-rep)
        output_file="$OUTPUT_DIR/${file_basename}"
        
        echo "Converting: $(basename "$nsys_file")"
        echo "  -> $output_file.prv"
        
        $SINGU_PREFIX  nsys2prv -t "$TRACE_TYPES" -m "$nsys_file" "$output_file"
        if [[ $? -eq 0 ]]; then
            echo "  [OK] Success"
            ((SUCCESS_COUNT++))
        else
            echo "  [FAIL] Conversion failed"
            ((FAIL_COUNT++))
        fi
        echo ""
    done
    
    echo "========================================"
    echo "Conversion Summary"
    echo "  Successful: $SUCCESS_COUNT"
    echo "  Failed:     $FAIL_COUNT"
    echo "  Total:      ${#NSYS_FILES[@]}"
    echo "========================================"
    
    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    fi
else
    # Run nsys2prv with all files at once (merged)
    echo "Running: singularity exec $CONTAINER nsys2prv -t $TRACE_TYPES -m ${NSYS_FILES[*]} $OUTPUT_PATH"
    echo ""
    
    $SINGU_PREFIX nsys2prv -t "$TRACE_TYPES" -m "${NSYS_FILES[@]}" "$OUTPUT_PATH"
    if [[ $? -eq 0 ]]; then
        echo ""
        echo "========================================"
        echo "[OK] Conversion successful"
        echo "Output: $OUTPUT_PATH.prv"
        echo "========================================"
    else
        echo ""
        echo "========================================"
        echo "[FAIL] Conversion failed"
        echo "========================================"
        exit 1
    fi
fi
