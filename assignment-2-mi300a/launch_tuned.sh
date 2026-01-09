#!/bin/bash

# script made for testing inside an interactive shell, do not use inside sbatch

export HIPBLASLT_LOG_LEVEL=0
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_FILENAME="/workspace/assignment-2/tuning/rocm7_tuned.csv"
export PYTORCH_TUNABLEOP_VERBOSE=1
export PYTORCH_TUNABLEOP_TUNING=0

python train.py --compile --fused-optimizer