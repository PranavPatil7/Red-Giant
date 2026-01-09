import torch
import torch.distributed as dist
import pandas as pd
import os

def create_local_tunables(rank, world_size, tunables_name):
    df = pd.read_csv(tunables_name)
    print(f"rank {rank} sees {len(df)} tunable gemms")
    local_subset_idx = [i for i in range(rank, len(df), world_size)]
    df_sub = df.iloc[local_subset_idx]
    df_sub.to_csv(f"/workspace/assignment-2/tuning/tunableop_untuned{rank}.csv", index=False)

def tune_gemms(rank):
    torch.cuda.tunable.enable(True)
    torch.cuda.tunable.tuning_enable(True)

    print(f"[tuner] rank: {rank} - Tuning start")

    torch.cuda.tunable.tune_gemm_in_file(f"/workspace/assignment-2/tuning/tunableop_untuned{rank}.csv")

    print(f"[tuner] rank: {rank} - Tuning end")

if __name__ == "__main__":

    untuned_file_name = "/workspace/assignment-2/tuning/to_tune.csv"
    # Initializes the default (global) process group
    dist.init_process_group(
        backend="nccl",  # the NCCL backend requires a GPU on each process
    )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # Limit GPU allocation of this process to only one GPU
    torch.cuda.set_device(local_rank)

    print(f"process {local_rank} in {rank}")

    create_local_tunables(rank, world_size, untuned_file_name)

    tune_gemms(rank)

    # Cleanup
    dist.destroy_process_group()
    