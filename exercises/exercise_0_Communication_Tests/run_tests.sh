#!/bin/bash
#SBATCH --job-name=0_comm_tests
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

export NODE_LIST=$SLURM_JOB_NODELIST
echo "Running on nodes: $NODE_LIST"
srun --export=ALL --nodes=$SLURM_NNODES --ntasks-per-node=1 bash $EXERCISE_DIR/tests.sh "$SLURM_JOB_NODELIST"