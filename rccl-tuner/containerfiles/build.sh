#!/bin/bash -eux 
set -o pipefail

extra_build_args=""
version () {
    extra_build_args="$extra_build_args --build-arg ${1}_VERSION=$2"
}

MI_TARGET='gfx942'

version MPI4PY '4.1.0'
version CUPY '13.6.0'
version RCCL '5bd5079'
version RCCL_TESTS 'd0a99b1'
version MPICH '4.2.3'
# version MPICH '3.4.3'
version CXI_PLUGIN '1ea938a'
version CXI_PLUGIN_AWS 'a3b1c71'
#version PMIX '3.2.3'
version PMIX '4.2.9'

version HWLOC 2.12
version HWLOC_PATCH 2
# version CASSINI_HEADERS 9a8a738a879f007849fbc69be8e3487a4abf0952
# version CXI_DRIVER c284516fa8d0027b8f84e8ca818e8c6bbe320bd8
# version LIBCXI 0f3609b5e224636abffc7ceffe6f4e9c83244c08
# version LIBFABRIC e08d3161b8038ebfa50a4eb5bf76132e4eaeb137

version CASSINI_HEADERS 4d2c6845ae99f70ebc4b96e6a82b20989cc6bfaf
version CXI_DRIVER e35fc1123fabf50cdf5cfd97f6eee02f375a4acc
version LIBCXI 77f876f337bfe8c118cd978b0e91bb988b9e165f
version LIBFABRIC 2b15467db18b1a204e7dcff1e91be3083fc06a75

version PYTHON '3.12'
version PYTORCH '2.8.0+rocm6.3'
version APEX '4b03581'
version TORCHVISION '0.23.0+rocm6.3'
version TORCHDATA '0.10.0'
version TORCHTEXT '0.18.0'
version TORCHAUDIO '2.8.0+rocm6.3'
version DEEPSPEED '0.17.4'
version FLASH_ATTENTION 'v2.8.3'
version VLLM 'v0.10.1'
version TE '1139bc1'
version MEGATRON '856c36d'
version AITER 'a2122a2'
version XFORMERS '90f6960'
version BITS_AND_BYTES '48a551f'
version AOTRITON '0.10b'

version TORCHAO '0.13.0+rocm6.3'
version TORCHTUNE 'v0.6.1'

# From AOTRITON deps -> triton -> llvm hash
version TRITON 'aotriton/0.10'
version TRITON_LLVM '3c709802d31b5bc5ed3af8284b40593ff39b9eec'

# Based on Pytorch commits: Check .ci/docker/ci_commit_pins/triton.txt in Pytorch
version TRITON_FINAL 96316ce50fade7e209553aba4898cd9b82aab83b
version TRITON_FINAL_LLVM a66376b0dc3b2ea8a84fda26faca287980986f78

#
# You should not need to change anything below this line.
#

BASENAME=rocm711
TAG=sebkelle-mi300a:$BASENAME
#SIF=$(basename $(pwd))-$BASENAME.sif
DOCKERFILE=build-$BASENAME.docker
LOG=build-$BASENAME.log

podman build \
  --build-arg MI_TARGET=$MI_TARGET \
  -f $DOCKERFILE \
  $extra_build_args \
  -t $TAG  2>&1 | tee $LOG

enroot import -x mount podman://$TAG -o $TAG.sqsh

#docker save $TAG -o /tmp/sfantao-sif-build-pipe/$SIF.tar

exit 0
