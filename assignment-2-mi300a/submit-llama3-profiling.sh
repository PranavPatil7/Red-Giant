#!/bin/bash

#SBATCH --account=a-g177-1
#SBATCH --time=00:20:00
#SBATCH --job-name=assignment_2_profiling
#SBATCH --output=/iopsstor/scratch/cscs/atazza/project/assignment-2/logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --partition=mi300
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --environment=/users/atazza/scratch/project/env.toml   # Vanilla 25.01 PyTorch NGC Image 
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

echo "START TIME: $(date)"

# Set up ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ASSIGNMENT_DIR="/iopsstor/scratch/cscs/atazza/project/assignment-2"

CMD_PREFIX="numactl --membind=0-3"

# TRAINING_CMD="python3 $ASSIGNMENT_DIR/train.py \
#     --batch-size 1 \
#     --learning-rate 5e-5 \
#     --lr-warmup-steps 100 \
#     --training-steps 1000 \
#     --sequence-length 4096 \
#     --compile \
#     --fused-optimizer
#     "

export ROC_PROFILER_MODE=1
export HIP_VISIBLE_DEVICES=0
export ROC_PROFILER_LAYER=all 

TRAINING_CMD="rocprof-sys-run --profile --trace --include ompt -- \
        python3 $ASSIGNMENT_DIR/train.py --profile
        "

srun --cpus-per-task=$SLURM_CPUS_PER_TASK bash -c "$TRAINING_CMD"

echo "END TIME: $(date)"