FROM rocm/dev-ubuntu-22.04:7.0.2

RUN apt-get update && \
    apt-get install -y \
        build-essential git python3-setuptools python3-wheel python3-pip python3-psutil \
        ninja-build libboost-filesystem-dev libboost-program-options-dev libboost-system-dev \
        vim hipblaslt llvm && \
    apt-get clean

RUN pip3 install --upgrade pip setuptools wheel \
    && pip3 install joblib packaging pyyaml cmake==3.28.1 msgpack tqdm
# workspace

RUN mkdir /workspace
WORKDIR /workspace

# clone ROCm libs
# RUN git clone -b rocm-6.4.1 --depth 1 https://github.com/ROCm/rocm-libraries.git

RUN mkdir -p /workspace/rocm-libraries/projects

# build rocblas first
RUN cd /workspace/rocm-libraries/projects/ && \
    git clone -b rocm-7.0.2 --depth 1 https://github.com/ROCm/rocBLAS.git && \
    cd rocBLAS && \
    ./install.sh -idc -a gfx942

# now build hipblaslt
RUN cd /workspace/rocm-libraries/projects/ && \
    git clone -b rocm-7.0.2 --depth 1 https://github.com/ROCm/hipBLASLt.git && \
    cd hipBLASLt && \
    export HIP_VISIBLE_DEVICES=0,1,2,3 && \
    ./install.sh -idc -a gfx942