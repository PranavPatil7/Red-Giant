#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --job-name=assignment-2-flash-attention-triton
#SBATCH --partition=mi300
#SBATCH --output=/iopsstor/scratch/cscs/atazza/project/assignment-2/logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --environment=/users/atazza/scratch/project/env.toml   
#SBATCH --no-requeue	
#SBATCH --container-writable

export ROCM_LOG_LEVEL=6
export ROCBLAS_LAYER=TRACE
export ROCBLAS_LAYER_LOG=1
export HIPBLASLT_LOG_LEVEL=6
export TRITON_KERNEL_DUMP=1
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
# export FLASH_ATTENTION_TRITON_AMD_AUTOTUNE="TRUE"

echo "START TIME: $(date)"

# Set up ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ASSIGNMENT_DIR="/iopsstor/scratch/cscs/atazza/project/assignment-2/"

# cd $ASSIGNMENT_DIR/flash-attention
# FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" pip install . --no-build-isolation

CMD_PREFIX="numactl --membind=0-3"

TRAINING_CMD="python3 $ASSIGNMENT_DIR/train.py \
    --batch-size 1 \
    --learning-rate 5e-5 \
    --lr-warmup-steps 100 \
    --training-steps 1000 \
    --sequence-length 4096 \
    --compile \
    --fused-optimizer
    "

srun --cpus-per-task=$SLURM_CPUS_PER_TASK bash -c "$CMD_PREFIX $TRAINING_CMD"

echo "END TIME: $(date)"
