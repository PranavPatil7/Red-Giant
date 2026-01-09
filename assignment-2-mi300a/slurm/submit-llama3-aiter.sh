#!/bin/bash

#SBATCH --time=00:50:00
#SBATCH --job-name=assignment-2-aiter
#SBATCH --partition=mi300
#SBATCH --output=/iopsstor/scratch/cscs/atazza/project/assignment-2/logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --environment=/iopsstor/scratch/cscs/atazza/project/aiter_env7.toml  
#SBATCH --no-requeue	
#SBATCH --container-writable

export TRITON_KERNEL_DUMP=1
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
# export FLASH_ATTENTION_TRITON_AMD_AUTOTUNE="TRUE" activate if triton flash attention is not tuned, it will cache it
export HIPBLASLT_LOG_LEVEL=0
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_FILENAME="/workspace/assignment-2/tuning/rocm7_tuned.csv"
export PYTORCH_TUNABLEOP_VERBOSE=1
export PYTORCH_TUNABLEOP_TUNING=0 # remember this flag, otherwise it will tune regardless of everything

echo "START TIME: $(date)"

# Set up ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ASSIGNMENT_DIR="/workspace/assignment-2/"

CMD_PREFIX="numactl --membind=0-3"

TRAINING_CMD="python3 $ASSIGNMENT_DIR/train.py \
    --batch-size 1 \
    --learning-rate 5e-5 \
    --lr-warmup-steps 100 \
    --training-steps 1000 \
    --sequence-length 4096 \
    --fused-optimizer \
    --compile
    "

srun --cpus-per-task=$SLURM_CPUS_PER_TASK bash -c "$CMD_PREFIX $TRAINING_CMD"

echo "END TIME: $(date)"
