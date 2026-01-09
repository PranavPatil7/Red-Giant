#!/bin/bash

#SBATCH --account=a-g177-1
#SBATCH --time=24:00:00
#SBATCH --job-name=podman_build
#SBATCH --partition=mi300
#SBATCH --output=/iopsstor/scratch/cscs/tschwab/Hackathon/hipblaslt/newlogs.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=192
#SBATCH --mem=512000
#SBATCH --no-requeue
#SBATCH --container-writable

podman build -t rocm/hipblaslt:7.0.2 .
enroot import -o rocm-hipblaslt-25_5.sqsh podman://rocm/hipblaslt:7.0.2