#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --partition=mi300
#SBATCH --time=1:00:00

set -x

python3 comm_tune.py