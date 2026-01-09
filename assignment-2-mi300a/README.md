# Single-GPU Performance Tuning on MI300

Here the code to obtain reasonable performance on assignment-2 on a single MI300A GPU can be found.
Some important observations:
- Communications are not relevant here, though they become a bottleneck when launching distributed jobs
- When using PyTorch to perform GEMMs, [Torch Tunable Ops](https://rocm.blogs.amd.com/artificial-intelligence/pytorch-tunableop/README.html) can be used to obtain much better performance
- Triton flash-attention on AMD MI300A performs consistetly better than aiter's implementation, though this might be investigated more
- Torch inductor max-autotune is an additional step to obtain better fused GEMM+activation kernels, in a very reasonable tuning time
