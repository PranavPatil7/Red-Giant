import os
import os.path as osp
from itertools import product
import subprocess

COLLECTIVES = {
    "allgather": ["all_gather_perf", ["ring"]],
    "reducescatter": ["reduce_scatter_perf", ["ring"]],
    "allreduce": ["all_reduce_perf", ["ring", "tree"]]
}

def tune_collective(coll_name, n_nodes, size_in_bytes):
    channels = [1, 2, 4, 8, 16, 32, 64, 128]
    proto = ["ll", "simple"]
    
    jobid = int(os.getenv("SLURM_JOB_ID"))

    rccl_bin, algo = COLLECTIVES[coll_name]

    n_ranks = n_nodes * 4

    os.makedirs("experiment", exist_ok=True)
    for c, p, a in product(channels, proto, algo):
        env_path = "/iopsstor/scratch/cscs/atazza/project/rccl.toml"
        plugin_path = "/iopsstor/scratch/cscs/atazza/project/rccl-tuner/libnccl-tuner.so"
        gen_conf = f"{coll_name},{size_in_bytes // 2},{size_in_bytes * 2 - 1},{a},{p},{c},{n_nodes},{n_ranks}"
        gen_conf_path = osp.join("/iopsstor/scratch/cscs/atazza/project/all_reduce_sweep/experiment",f"tuning_{coll_name}_{size_in_bytes}_{c}_{p}_{a}_{n_nodes}_{n_ranks}_{jobid}.conf")
        result_path = osp.join("/iopsstor/scratch/cscs/atazza/project/all_reduce_sweep/experiment", f"tuning_{coll_name}_{size_in_bytes}_{c}_{p}_{a}_{n_nodes}_{n_ranks}_{jobid}.csv")
        with open(gen_conf_path,"w") as f:
            f.write(gen_conf)
        cmd_options = f"-b {size_in_bytes // 2} -e {size_in_bytes*2} -f 2 -g 1 -M 1 -Z csv -x {result_path}"
        cmd = f"export NCCL_DEBUG=TRACE\nexport NCCL_DEBUG_SUBSYS=TUNING\nexport NCCL_TUNER_PLUGIN={plugin_path}\nexport NCCL_TUNER_CONFIG_FILE={gen_conf_path}\nsrun --output=logs/{coll_name}_%j.log -p mi300 -N{n_nodes} --ntasks-per-node=4 --hint=nomultithread --mpi=pmi2 -c24 --environment={env_path} /opt/rccltests/{rccl_bin} {cmd_options}"
        subprocess.run(cmd, shell=True)
        print(f"{c,p,a} done")

if __name__ == "__main__":
    import os
    tune_collective(
        "reducescatter",
        int(os.getenv("SLURM_NNODES")),
        2**27
    )