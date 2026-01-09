FROM rocm/dev-ubuntu-22.04:7.0.2

ARG ROCM_VERSION=7.0.2
ARG AMDGPU_VERSION=7.0.2.70002
ARG PYTHON_VERSION=3.12

# check https://repo.radeon.com/rocm/manylinux/ for possible versions, select appropriate rocm version, check for matching python version
ARG PYTORCH_VERSION=2.8.0
ARG ROCM_VERSION_FULL=7.0.2.git245bf6ed

# HipBLASLt repo and branch (custom), TODO: change to if ROCM VERSION >= 7.0 rocm-libraries else legacy repo
ARG HIPBLASLT_REPO=https://github.com/ROCm/rocm-libraries.git
ARG HIPBLASLT_BRANCH=rocm-${ROCM_VERSION}
# ARG HIPBLASLT_BRANCH=634088c98d

# Flashattention repo and branch (AMD)
ARG FA_REPO=https://github.com/ROCm/flash-attention.git
ARG FA_BRANCH=6387433156558135a998d5568a9d74c1778666d8
# ARG FA_BRANCH=6387433156558135a998d5568a9d74c1778666d8

# TensorEngine repo and branch (custom)
ARG TE_REPO=https://github.com/ROCm/TransformerEngine.git
ARG TE_BRANCH=v2.1_rocm
# ARG TE_BRANCH=c3bcaab1311d3d961880733eeab87ea7391adb99

# aiter repo and branch (from AMD)
ARG AITER_REPO=https://github.com/rocm/aiter.git
ARG AITER_BRANCH=main

# AMD Diffusion Benchmark repo and branch (from AMD)
ARG FLUX_REPO=https://github.com/ROCm/AMDiffusionBenchmark.git
ARG FLUX_BRANCH=Flux_patch

# torchtitan repo and branch (from AMD)
ARG TORCHTITAN_REPO=https://github.com/AMD-AIG-AIMA/torchtitan.git
ARG TORCHTITAN_BRANCH=dev/aiter_attn_random_input

ARG TORCHTUNE_REPO=https://github.com/pytorch/torchtune.git
ARG TORCHTUNE_BRANCH=b4c98ac2a37f0397d64c22579aed415ce7264db6

ARG TORCHAO_REPO=https://github.com/pytorch/ao.git
ARG TORCHAO_BRANCH=a9ffa508ef191eb69119658827cdda90dd918dca

ARG GROUPED_GEMM_REPO=https://github.com/caaatch22/grouped_gemm.git
ARG GROUPED_GEMM_BRANCH=rocm

ARG CAUSAL_CONV1D_REPO=https://github.com/Dao-AILab/causal-conv1d
ARG CAUSAL_CONV1D_BRANCH=e940ead2fd962c56854455017541384909ca669f

ARG ROCSHMEM_REPO=https://github.com/ROCm/rocSHMEM.git
ARG ROCSHMEM_BRANCH=e95360961dfa5575a49fe79abdf41ab9e4aeb543

# reframe tests use custom Megatron repo
# ARG MEGATRON_REPO=https://github.com/ROCm/Megatron-LM.git
# ARG MEGATRON_VERSION=

# custom triton Version suffix and branch
ARG TRITON_REPO=https://github.com/ROCm/triton.git
ARG TRITON_BRANCH=1cc4a7c4aac921b7fa4658e162058632b7faed55
ARG TRITON_VERSION_SUFFIX="+rocm7.0.2.gita272dfa8"
# ARG TRITON_VERSION_SUFFIX="+rocm7.0.0.git56765e8c"

ARG TRITON_VERSION=3.5.1

ENV MAX_JOBS=128
ENV PYTORCH_ROCM_ARCH=gfx942
ENV GPU_ARCHS=gfx942

ENV NVTE_USE_HIPBLASLT=1
ENV NVTE_FRAMEWORK=pytorch
ENV NVTE_ROCM_ARCH=gfx942
ENV NVTE_USE_CAST_TRANSPOSE_TRITON=0
ENV NVTE_CK_IS_V3_ATOMIC_FP32=0
ENV NVTE_CK_USES_BWD_V3=1
ENV NVTE_CK_USES_FWD_V3=1
ENV CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=2
ENV NVTE_USE_ROCM=1
ENV CU_NUM=228

ENV AITER_ASM_DIR=/workspace/TransformerEngine/3rdparty/aiter/hsa/gfx942/

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y wget software-properties-common curl rsync dialog git

ENV PATH=/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/opt/rocm/lib

# Python
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && apt install -y python${PYTHON_VERSION} python3-pip python3-venv && \
    if [ "${PYTHON_VERSION}" = "3.8" ] || [ "${PYTHON_VERSION}" = "3.9" ] || [ "${PYTHON_VERSION}" = "3.10" ] || [ "${PYTHON_VERSION}" = "3.11" ]; then \
        apt-get install -y --no-install-recommends \
            python${PYTHON_VERSION}-dev \
            python${PYTHON_VERSION}-distutils \
            python${PYTHON_VERSION}-venv; \
    else \
        apt-get install -y --no-install-recommends \
            python${PYTHON_VERSION}-dev \
            python${PYTHON_VERSION}-venv; \
    fi && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# create venv
RUN python${PYTHON_VERSION} -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install --upgrade setuptools


ENV PATH=/opt/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN wget https://repo.radeon.com/amdgpu-install/${ROCM_VERSION}/ubuntu/jammy/amdgpu-install_${AMDGPU_VERSION}-1_all.deb

# RUN apt install -y ./*.deb

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    -o Dpkg::Options::="--force-confnew" \
    ./*.deb

RUN amdgpu-install --usecase=rocm -y && rm *.deb

# PyTorch
# RUN set -e && \
#     echo "Installing PyTorch via .sh installer for ROCm $ROCM_VERSION, Torch $PYTORCH_VERSION, Python $PYTHON_VERSION" && \
#     if [ "$PYTHON_VERSION" = "3.12" ]; then \
#         PYTHON_TAG="cp312-cp312"; \
#     elif [ "$PYTHON_VERSION" = "3.10" ]; then \
#         PYTHON_TAG="cp310-cp310"; \ 
#     else \ 
#         echo "Unsupported Python version: $PYTHON_VERSION"; exit 1; \
#     fi && \
#     TORCH_SH=$(curl -fsSL https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_VERSION}/.sh-scripts | grep "torch-${PYTORCH_VERSION}%2Brocm${ROCM_VERSION_FULL}.lw" | grep "${PYTHON_TAG}" | grep "\.sh" | sed -n 's/.*href="\([^"]*\).*/\1/p' | head -n1) && \
#     if [ -z "$TORCH_SH" ]; then \
#         echo "No matching torch .sh installer found for torch-${PYTORCH_VERSION}+rocm${ROCM_VERSION_FULL} (${PYTHON_TAG})"; exit 1; \
#     fi && \ 
#     echo "Found torch installer: $TORCH_SH" && \
#     mkdir -p /install && \
#     curl -fsSL -o /install/torch_installer.sh https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_VERSION}/.sh-scripts/${TORCH_SH} && \
#     chmod +x /install/torch_installer.sh && \
#     sh /install/torch_installer.sh

RUN set -e && \
    echo "Installing PyTorch via .whl for ROCm $ROCM_VERSION, Torch $PYTORCH_VERSION, Python $PYTHON_VERSION" && \
    if [ "$PYTHON_VERSION" = "3.12" ]; then \
        PYTHON_TAG="cp312-cp312"; \
    elif [ "$PYTHON_VERSION" = "3.10" ]; then \
        PYTHON_TAG="cp310-cp310"; \
    else \
        echo "Unsupported Python version: $PYTHON_VERSION"; exit 1; \
    fi && \
    TORCH_WHL=torch-${PYTORCH_VERSION}%2Brocm${ROCM_VERSION_FULL}-${PYTHON_TAG}-linux_x86_64.whl && \
    TORCHAUDIO_WHL=torchaudio-2.8.0%2Brocm7.0.2.git6e1c7fe9-cp312-cp312-linux_x86_64.whl && \
    TORCHVISION_WHL=torchvision-0.24.0%2Brocm7.0.2.gitb919bd0c-cp312-cp312-linux_x86_64.whl && \
    TRITON_WHL=triton-3.4.0%2Brocm7.0.2.gitf9e5bf54-cp312-cp312-linux_x86_64.whl && \
    APEX_WHL=apex-1.9.0a0%2Brocm7.0.2.git07c3ee53-cp312-cp312-linux_x86_64.whl && \
    wget https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_VERSION}/${TORCH_WHL} && \
    wget https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_VERSION}/${TORCHAUDIO_WHL} && \
    wget https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_VERSION}/${TORCHVISION_WHL} && \
    wget https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_VERSION}/${TRITON_WHL} && \
    wget https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_VERSION}/${APEX_WHL} && \
    pip install --break-system-packages *.whl

RUN git clone https://github.com/ROCm/pytorch-micro-benchmarking /var/lib/jenkins/pytorch-micro-benchmarking

ENV ROCM_PATH=/opt/rocm-${ROCM_VERSION}

RUN mkdir -p workspace
WORKDIR /workspace/

RUN apt --fix-broken -y install
RUN pip install pybind11 typeguard

ENV HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0

# COPY /*.whl build/

# RUN pip install build/*.whl

# RUN rm build/*

# get specific cmake version
RUN pip uninstall -y cmake && \
    pip install cmake==3.31.6

# get specific ninja version
RUN pip uninstall -y ninja && \
    pip install ninja==1.11.1.3

# other dependencies
RUN apt install -y ninja-build autoconf libtool flex rdma-core
RUN cd /opt/rocm/share/amd_smi && pip install . && \
    cd -

# HipBLASLt
RUN apt-get --purge remove -y hipblaslt hipblaslt-dev && \
    rm -fr /opt/rocm/include/hipblaslt && \
    rm -fr /opt/rocm/lib/libhipblaslt* && \
    rm -rf /opt/rocm-${ROCM_VERSION}/lib/hipblaslt && \
    git clone ${HIPBLASLT_REPO} && \
    cd rocm-libraries/projects/hipblaslt && \
    git checkout ${HIPBLASLT_BRANCH} && \
    # git cherry-pick e0deaed2fa081e904b19e09d73257c1d4a817e99 --strategy-option theirs || true && \
    # git cherry-pick 5e2e486d6568b1b8b9f1656edae75a3fe45288a1 --strategy-option theirs || true && \
    apt-get update && apt-get install -y libboost-filesystem-dev && \
    python -m pip install -r tensilelite/requirements.txt && \
    MAX_JOBS=${MAX_JOBS} \
    ./install.sh -idc -a ${PYTORCH_ROCM_ARCH} && \
    cd ../../../

RUN ln -s /opt/rocm/lib/libhipblaslt.so /opt/rocm/lib/libhipblaslt.so.0

RUN apt-get install -y miopen-hip rocsolver hipblas hipsolver

# flashattention
RUN git clone --recursive ${FA_REPO} && \
    cd flash-attention && \
    git checkout ${FA_BRANCH} && \
    python setup.py install && \
    cd .. && \
    rm -rf flash-attention

# TranformerEngine
RUN git clone --recursive ${TE_REPO} && \
    cd TransformerEngine && \
    git checkout ${TE_BRANCH} && \
    git submodule update --init --recursive && \
    MAX_JOBS=${MAX_JOBS} pip install . && \
    cd ..

# aiter
RUN git clone --recursive ${AITER_REPO} && \
    cd aiter && \
    git checkout ${AITER_BRANCH} && \
    find aiter/ops/mha.py -type f -print0 | xargs -0 sed -i 's/is_v3_atomic_fp32: Optional\[bool\] = True,/is_v3_atomic_fp32: Optional\[bool\] = False,/g' && \
    python setup.py develop && \
    cd ..

# AMD Diffusion Benchmark
RUN git clone ${FLUX_REPO} && \
    cd AMDiffusionBenchmark && \
    git checkout ${FLUX_BRANCH} && \ 
    pip install -r requirements.txt && \
    cd ..

# torchtitan
RUN git clone ${TORCHTITAN_REPO} && \
    cd torchtitan && \
    git checkout ${TORCHTITAN_BRANCH} && \
    pip install -r requirements.txt && \
    pip install -e . && \
    cd ..

# install torchtune
RUN git clone ${TORCHTUNE_REPO} && \
    if [ "${TORCHTUNE_BRANCH}" != "" ]; then \
        cd torchtune && \
        git checkout ${TORCHTUNE_BRANCH} && \
        cd ..; \
    fi && \
    cd torchtune && \
    git checkout ${TORCHTUNE_BRANCH} && \
    find torchtune/modules/moe/utils.py -type f -print0 | xargs -0 sed -i 's/use_grouped_mm = True/use_grouped_mm = False/g' && \
    pip install -e . && \
    pip install numpy==1.26.4 && \
    cd ..

# Torch/ao
RUN bin/sh -c git clone ${TORCHAO_REPO} && \
    cd ao && git checkout ${TORCHAO_BRANCH} && \
    find torchao/float8/config.py -type f -print0 | xargs -0 sed -i 's/pad_inner_dim: bool = False/pad_inner_dim: bool = True/g' && \
    find torchao/csrc/rocm/swizzle/swizzle.cpp -type f -print0 | xargs -0 sed -i 's/if defined(HIPBLASLT_VEC_EXT)/if false/g' && \
    pip install . && \
    cd .. && \
    rm -r ao

RUN pip install datasets numpy==1.26.4 transformers==4.55.0
RUN pip install --upgrade 'optree>=0.13.0'
RUN pip install --upgrade sympy
RUN pip install accelerate==1.9.0 trl==0.21.0 peft
RUN pip3 install scipy einops flask-restful nltk pytest pytest-cov pytest_mock pytest-csv pytest-random-order sentencepiece \
                            wrapt zarr wandb tensorstore==0.1.45 pytest_mock pybind11 setuptools==69.5.1 datasets tiktoken pynvml
RUN pip3 install "huggingface_hub[cli]"
RUN nltk.downloader punkt_tab

RUN git clone ${GROUPED_GEMM_REPO} && \
    cd grouped_gemm && \
    git checkout ${GROUPED_GEMM_BRANCH} && \
    git submodule update --init --recursive && \
    pip install . && \
    cd ../ && \
    rm -rf grouped_gemm

ENV CAUSAL_CONV1D_FORCE_BUILD=TRUE
ENV MAMBA_FORCE_BUILD=TRUE
ENV HIP_ARCHITECTURES=gfx942

RUN git clone ${CAUSAL_CONV1D_REPO} causal-conv1d && \
    cd causal-conv1d && \
    git checkout ${CAUSAL_CONV1D_BRANCH} && \
    git show --oneline -s && \
    pip install . && \
    cd ../ && \
    rm -fr causal-conv1d

RUN ulimit -n 8192 && \
    git clone ${ROCSHMEM_REPO} && \
    cd rocSHMEM && \
    git checkout ${ROCSHMEM_BRANCH} && \
    find scripts/install_dependencies.sh -type f -print0 | xargs -0 sed -i 's/set -o pipefail//g' && \
    find scripts/install_dependencies.sh -type f -print0 | xargs -0 sed -i 's/export _INSTALL_DIR=\$BUILD_DIR\/install/export _INSTALL_DIR=\/opt/g' && \
    ./scripts/install_dependencies.sh && cd .. && rm -rf rocSHMEM

RUN git clone ${TRITON_REPO} && \
    cd triton && git checkout ${TRITON_BRANCH} && \
    pip install -r python/requirements.txt && \
    TRITON_WHEEL_VERSION_SUFFIX=${TRITON_VERSION_SUFFIX} pip install . && \
    cd .. && \
    rm -rf triton

ARG MEGATRON_REPO='https://github.com/ROCm/Megatron-LM.git'
ARG MEGATRON_VERSION='856c36d'
RUN  set -eux ; \
  rm -rf /opt/mybuild ;\
  git clone $MEGATRON_REPO /opt/mybuild ; \
  cd /opt/mybuild ; \
  git checkout -b mydev $MEGATRON_VERSION ; \
  git submodule sync ; \
  git submodule update --init --recursive  ; \
    python3 setup.py bdist_wheel --dist-dir=/opt/wheels ; \
  cd / ; rm -rf /opt/mybuild

RUN  set -eux ; \
  pip install --break-system-packages /opt/wheels/megatron_core-*.whl ; \
  true