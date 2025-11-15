#!/bin/bash

module purge
module unload compiler/rocm/2.9
module load compiler/rocm/dtk-23.04
#module load compiler/rocm/dtk-22.10
module load compiler/devtoolset/7.3.1
module load apps/tmux/gcc-7.3.1
#module load mpi/hpcx/2.6.0/gcc-7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1

#export NCCL_DEBUG=INFO
export NCCL_DEBUG=DEBUG
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=eno,en,eth,em,bond,ib0,eno1
export NCCL_IB_DISABLE=1

#export LD_LIBRARY_PATH=/public/home/chenhao/.conda/envs/llm-test/lib/python3.9:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/chenhao/.conda/envs/llm-test/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH
#source activate llm-test2

echo "test cuda"
python -c "import torch; print(torch.cuda.is_available())"


export NCCL_NET_GDR_LEVEL=SYS
export NCCL_P2P_LEVEL=5

export MIOPEN_FIND_MODE=3
export HSA_FORCE_FINE_GRAIN_PCIE=1
export MIOPEN_COMPILE_PARALLEL_LEVEL=1
export RCCL_NCHANNELS=2

export MODEL_SIZE=13
export SEQ_LEN=4096
export GC_SCALE=4
export DATE=20230322
export USE_FLASH_ATTN=0


