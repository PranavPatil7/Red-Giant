#!/bin/bash

#SBATCH --account=a-g177-1
#SBATCH --time=24:00:00
#SBATCH --job-name=podman_build
#SBATCH --partition=mi300
#SBATCH --output=/iopsstor/scratch/cscs/tschwab/Hackathon/custom-env/logs.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=192
#SBATCH --mem=512000
#SBATCH --no-requeue
#SBATCH --container-writable

podman build -t custom_megatronlm:7.0.2 .
enroot import -o custom_megatronlm_7_0_2.sqsh podman://custom_megatronlm:7.0.2