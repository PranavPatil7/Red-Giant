To run reframe tests for LLMs on CSCS Beverin, first start by cloning the official reframe repo into the "Large-Scale AI" repository

```bash
git clone https://github.com/reframe-hpc/reframe.git
pushd reframe
git checkout 07ff6b30d7be8aadbff0433d171d9e7f1403b8a8
./bootstrap.sh
export PATH=$(pwd)/bin:$PATH
popd
```

The MegatronLM test for Llama3.1-8b can be run using 

```bash
cd cscs-reframe-tests
reframe \
-C config/cscs.py \
-c checks/apps/pytorch/pytorch_megatronlm_amd.py --run
```


### Performance Guidelines
During this project, we identified two major bottlenecks for out-of-the-box performance: collective communication protocols and GEMM kernel selection. Collectives play a major role in distributed training; the most frequently used communication patterns are AllGather and ReduceScatter. Tuning for collective communication refers to selecting an appropriate protocol for a given buffer size, communication pattern, and cluster network topology. Part of the results were deduced empirically. During the Hackathon in Lugano, we found the following environment variables to work best
```python
'NCCL_CROSS_NIC': 1,
'NCCL_NET': '"AWS Libfabric"',
'NCCL_NET_GDR_LEVEL': 'PHB',
'NCCL_PROTO': '^LL128',
'FI_CXI_DEFAULT_CQ_SIZE': 131072,
'FI_CXI_DEFAULT_TX_SIZE': 32768,
'FI_CXI_DISABLE_HOST_REGISTER': 1,
'FI_MR_CACHE_MONITOR': 'userfaultfd',
'FI_CXI_RDZV_EAGER_SIZE': 0,
'FI_CXI_RDZV_GET_MIN': 0,
'FI_CXI_RDZV_THRESHOLD': 0,
'HSA_NO_SCRATCH_RECLAIM': 1
```

We continue with a description of RCCL offline tuning.
First, we need to determine the sizes of the ReduceScatter and AllGather buffers. To do this, run the reframe test with NCCL Debug enabled. You will find outputs like

```bash
nid002664:106865:108086 [0] NCCL INFO ReduceScatter: 302006272 Bytes -> Algo 1 proto 2 time 3628.243652
nid002664:106865:106865 [0] NCCL INFO AllGather: 151003136 Bytes -> Algo 1 proto 2 time 0.000000
```
These are the sizes for the ReduceScatter and AllGather that should be tuned using the RCCL Tuner plugin.

See [here](./rccl-tuner/tuning.md) for more information on how to tune gemm shapes.

### Future proposed workflow
We have experienced issues with AMD's official container images; newer versions have been completely incompatible with CSCS's reframe test workflow or yielded below 10% of peak performance. This may be because the performing container images are based on Python 3.12 (rocm/megatron-lm:v25.5_py312 and rocm/megatron-lm:v25.6_py312), where newer versions are all based on Python 3.10. Additionally, we experienced issues with later RCCL versions.
Ideally, one would like to build the container from scratch to control dependencies, such as Python/PyTorch versions and the Megatron base repository, especially if one wants to train Apertus with its custom xielu activations on the Swiss AI fork of Megatron-LM.

#### Container Image
Begin by building a container using the Dockerfile provided in custom-env. Note that dependencies may vary between different ROCm versions. (Unfortunately, the Dockerfile has not been tested past the hipblaslt installation, so there is no guarantee that the version combination works.) If AMD does not provide any install scripts for PyTorch and Related libraries, you can find the build wheels on 
```
https://repo.radeon.com/rocm/manylinux/rocm-rel-X.Y.Z/
```

Where X.Y.Z is the desired ROCm base version of the build. Important are
- torch
- torchaudio
- torchvision
- triton
- apex

An example of the relevant wheels in the case of ROCm 7.0.2 and Python 3.12 is the following
```python
TORCH_WHL=torch-2.8.0%2Brocm7.0.2.lw.git245bf6ed-cp312-cp312-linux_x86_64.whl && \
TORCHAUDIO_WHL=torchaudio-2.8.0%2Brocm7.0.2.git6e1c7fe9-cp312-cp312-linux_x86_64.whl && \
TORCHVISION_WHL=torchvision-0.24.0%2Brocm7.0.2.gitb919bd0c-cp312-cp312-linux_x86_64.whl && \
TRITON_WHL=triton-3.4.0%2Brocm7.0.2.gitf9e5bf54-cp312-cp312-linux_x86_64.whl && \
APEX_WHL=apex-1.9.0a0%2Brocm7.0.2.git07c3ee53-cp312-cp312-linux_x86_64.whl
```

Currently, GEMMs are tuned automatically during the first training iteration. This is slow and inefficient, as tuning has to be done every time training is launched, and the first iteration is conducted using the untuned GEMMs. On AMD, the heuristics for selecting a GEMM are flawed, resulting in extremely poor throughput in the first iteration.

#### GEMM Offline Tuning
This motivates tuning the GEMMs offline. To get the most useful results, it should be done in the same environment as the training. When installing hipBLASLt in the Docker build, we do not delete the repository. This is to keep the clients and full offline tuning capability.
To find the GEMM shapes that are most dominant during training, we have to analyze the trace.

To obtain the trace of the GEMM shapes, use the following:
```bash
pip install perfetto # if not already installed
python3 ./tuning/find_gemm_shapes.py -f <path_to_perfetto_trace.json>
```
An example generated output (Llama3.1-8B) can be found [here](./tuning/gemm_shapes.csv)  

Now that we have found the correct shapes to tune, we can navigate to the hipblaslt repository and find the tuning script "find_exact.py" in the utilities folder. Before running it, we need to configure a yaml file 

```yaml 
Bench: 
	ProblemType:
		ComputeDataType: s 
		ComputeInputDataType: b
		DataTypeA: b 
		DataTypeB: b 
		DataTypeC: b 
		DataTypeD: b 
		TransposeA: 0 
		TransposeB: 0 
		UseBias: False 
	TestConfig: 
		ColdIter: 10 
		Iter: 50
		AlgoMethod: "all" # Fixed value 
		RotatingBuffer: 512 # It's recommended to set this value larger than the cache size of the GPU. 
	TuningParameters: # SplitK list control parameter example 
		SplitK: [0] # [0] For disable, or specify values like [0, 4, 8] 
	ProblemSizes: # Format: [m, n, batch_count, k] 
		- [14336, 4096, 1, 8192] 
		- [4096, 14336, 1, 8192] 
	    - [8192, 14336, 1, 4096] 
	    - [8192, 4096, 1, 6144] 
	    - [8192, 4096, 1, 14336] 
	    - [8192, 4096, 1, 4096] 
	    - [6144, 4096, 1, 8192] 
		- [4096, 4096, 1, 8192] 
		- [8192, 4096, 1, 32000] 
		- [32000, 4096, 1, 8192] 
		- [8192, 6144, 1, 4096]
		- [8192, 32000, 1, 4096] 
	# CreateLogic: {} # Uncomment this line when you want to create logic files after benchmarking
```
In ProblemSizes, we specify the array of GEMM shapes found by analyzing the trace.
Run the tuning utility inside the container using 
```bash
python3 find_exact.py <your yaml file> <hipblaslt_root_folder>/build/release <output folder>
```
Make sure to mount the output folder in the environment file, e.g., on beverin
```toml
image = <squash filesystem>

mounts = ["/capstor", "/iopsstor", "/users"]

writable = true
```
In future work, a way should be proposed to consistently load the gemm config for the kernels launched in TE. This may require a transformation of format from tuning output to TE load file.
In ROCm versions 6.3-6.4 we experienced issues with loading the configs the solution of which are beyond the scope of this project. Later versions seemed to have this functionality working, but broke in different ways. Principally, the ability to load and tune configs is implemented via
```python
if self.gemm_tuning:
    self.env_vars['TE_HIPBLASLT_TUNING_RUN_COUNT'] = 10
    self.env_vars['TE_HIPBLASLT_TUNING_ALGO_COUNT'] = 100
    if self.save_tuning_config:
        self.env_vars['TE_HIPBLASLT_ALGO_SAVE'] = "/path/to/results.csv"
else if self.load_tuning_config:
    self.env_vars['TE_HIPBLASLT_ALGO_LOAD'] = "/path/to/tuning.csv"
```

### Possible Performance Increase

The following is a comparison of the fastest observed GEMMs during training and tuning of hipBLASLt (branches rocm-6.3.3 and rocm-7.1.1) for the GEMM shapes encountered when working with Llama3.1-8b:
[![GEMM Performance](tuning/results/GEMM.png)](tuning/results/GEMM.pdf)