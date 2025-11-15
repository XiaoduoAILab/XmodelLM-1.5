#!/bin/bash

echo "node env prepare: $(date)"
source ./prepare_env.sh
echo "python env location: $(which python)"

export HSA_USERPTR_FOR_PAGED_MEM=0
export OMP_NUM_THREADS=4
export HSA_FORCE_FINE_GRAIN_PCIE=1
#export NCCL_DEBUG=INFO
export NCCL_DEBUG=DEBUG
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=en,eth,em,bond,ib
export NCCL_DEBUG_SUBSYS=ALL

lrank=$OMPI_COMM_WORLD_LOCAL_RANK
world_rank=$OMPI_COMM_WORLD_RANK
world_size=$OMPI_COMM_WORLD_SIZE
comm_node_rank=$OMPI_COMM_WORLD_NODE_RANK
comm_local_size=$OMPI_COMM_WORLD_LOCAL_SIZE
uni_size=$OMPI_UNIVERSE_SIZE
job_id=$SLURM_JOB_ID

GPUS_PER_NODE=4

ip=`ifconfig |grep inet | grep 10. |grep -v 127.0.0.1|grep -v inet6 |awk '{print $2}'|tr -d 'addr:'`
echo "client node ip is : $ip"

echo "lrank: ${lrank}"
echo "world_rank: ${world_rank}"
echo "world_size: ${world_size}"
echo "comm_node_rank: ${comm_node_rank}"
echo "comm_local_size: ${comm_local_size}"
echo "uni_size: ${uni_size}"
let node_size=`expr $uni_size/$GPUS_PER_NODE`
echo node_size $node_size
echo "job_id" $job_id
echo "GPUS_PER_NODE" $GPUS_PER_NODE

echo current_path_is `pwd`
echo "sh_path" `pwd`
#cd /public/home/chenhao/llm_test/llm_pretrain_xmodel/
echo "new_path" `pwd`

#校验master node ip写入是否正确
master_ip=`sed -n '1p' master_node_ip.txt`
echo "single_p master node ip is : $master_ip"


###### openwebtext torchrun A line
#APP_common="python -m torch.distributed.run --nnodes=${node_size} --nproc_per_node=4 --node_rank=${world_rank} --master_addr=${master_ip} --master_port=8822 pretrain/openwebtext_nscc.py "
#special_args=" --model_name nano --dtype float32 --total_steps 10000 --micro_batch_size 6 --gradient_accumulation_steps 3 --block_size 2048 --eval_interval 100 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name torchrun --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size} --line_index a_line/ "
#special_args=" --model_name micro --dtype float32 --total_steps 10000 --micro_batch_size 4 --gradient_accumulation_steps 4 --block_size 2048 --eval_interval 100 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name torchrun --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size} --line_index a_line/ "
#special_args=" --model_name tiny --dtype float32 --total_steps 10000 --micro_batch_size 2 --gradient_accumulation_steps 8 --block_size 2048 --eval_interval 100 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name torchrun --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size} --line_index a_line/ "
#special_args=" --model_name small --dtype float32 --total_steps 10 --micro_batch_size 1 --gradient_accumulation_steps 8 --block_size 2048 --eval_interval 10 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name torchrun --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size} --line_index a_line/ "
#special_args=" --model_name medium --dtype float32 --total_steps 10000 --micro_batch_size 1 --gradient_accumulation_steps 16 --block_size 2048 --eval_interval 100 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name torchrun --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size} --line_index a_line/ "
#special_args=" --model_name large --dtype float32 --total_steps 10000 --micro_batch_size 1 --gradient_accumulation_steps 16 --block_size 2048 --eval_interval 100 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name torchrun --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size} --line_index a_line/ "


###### openwebtext deepspeed launch openwebtext_deepspeed_nscc.py   B line
#APP_common="python -m torch.distributed.run --nnodes=${node_size} --nproc_per_node=4 --node_rank=${world_rank} --master_addr=${master_ip} --master_port=8822 pretrain/openwebtext_deepspeed_nscc.py "
#special_args=" --model_name nano --dtype float32 --total_steps 100 --micro_batch_size 2 --gradient_accumulation_steps 2 --max_length 2028 --eval_interval 20 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name deepspeed --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size} --line_index b_line/"
#special_args=" --model_name micro --dtype float32 --total_steps 100 --micro_batch_size 2 --gradient_accumulation_steps 2 --max_length 2028 --eval_interval 20 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name deepspeed --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size} --line_index b_line/"
#special_args=" --model_name tiny --dtype float32 --total_steps 60 --micro_batch_size 1 --gradient_accumulation_steps 2 --max_length 2028 --eval_interval 10 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name deepspeed --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size} --line_index b_line/"
#special_args=" --model_name small --dtype float32 --total_steps 60 --micro_batch_size 1 --gradient_accumulation_steps 2 --max_length 2028 --eval_interval 10 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name deepspeed --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size} --line_index b_line/"
#special_args=" --model_name medium --dtype float32 --total_steps 60 --micro_batch_size 1 --gradient_accumulation_steps 2 --max_length 2028 --eval_interval 10 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name deepspeed --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size} --line_index b_line/"
#special_args=" --model_name large --dtype float32 --total_steps 60 --micro_batch_size 1 --gradient_accumulation_steps 2 --max_length 2028 --eval_interval 10 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name deepspeed --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size} --line_index b_line/"
#APP_common="python pretrain/openwebtext_deepspeed_nscc.py "
#special_args=" --model_name nano --dtype float32 --total_steps 10 --micro_batch_size 2 --gradient_accumulation_steps 2 --max_length 2028 --eval_interval 10 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name deepspeed --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size} --line_index b_line/"
#special_args=" --model_name small --dtype float32 --total_steps 10 --micro_batch_size 1 --gradient_accumulation_steps 2 --max_length 2028 --eval_interval 10 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name deepspeed --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size} --line_index b_line/"
#special_args=" --model_name medium --dtype float32 --total_steps 10 --micro_batch_size 1 --gradient_accumulation_steps 2 --max_length 2028 --eval_interval 10 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name deepspeed --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size} --line_index b_line/"
#special_args=" --model_name large --dtype float32 --total_steps 10 --micro_batch_size 1 --gradient_accumulation_steps 2 --max_length 2028 --eval_interval 10 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name deepspeed --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size} --line_index b_line/"


###### redpajama torchrun C line
#APP_common="python -m torch.distributed.run --nnodes=${node_size} --nproc_per_node=4 --node_rank=${world_rank} --master_addr=${master_ip} --master_port=8822 pretrain/redpajama_nscc.py "
#special_args=" --model_name nano --dtype float32 --total_steps 350000 --micro_batch_size 1 --gradient_accumulation_steps 4 --max_length 4096 --eval_interval 2000 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name torchrun --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size} --line_index c_line/ --sampling random"
#special_args=" --model_name micro --dtype float32 --total_steps 350000 --micro_batch_size 1 --gradient_accumulation_steps 4 --max_length 4096 --eval_interval 2000 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name torchrun --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size} --line_index c_line/ --sampling random --resume"
#special_args=" --model_name tiny --dtype float32 --total_steps 350000 --micro_batch_size 1 --gradient_accumulation_steps 4 --max_length 4096 --eval_interval 2000 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name torchrun --initializer_range 0.02 --optimizer_eps 1e-8 --job_id 54482266 --node_size ${node_size} --line_index c_line/ --sampling random --resume"
#special_args=" --model_name small --dtype float32 --total_steps 350000 --micro_batch_size 1 --gradient_accumulation_steps 2 --max_length 4096 --eval_interval 2000 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name torchrun --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size} --line_index c_line/ "
#special_args=" --model_name medium --dtype float32 --total_steps 10 --micro_batch_size 1 --gradient_accumulation_steps 2 --max_length 4096 --eval_interval 10 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name torchrun --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size} --line_index c_line/ "



###### redpajama torchrun E line
#APP_common="python -m torch.distributed.run --nnodes=${node_size} --nproc_per_node=4 --node_rank=${world_rank} --master_addr=${master_ip} --master_port=8822 pretrain/train_e_line_nscc.py "
#special_args=" --model_name nano --dtype float32 --total_steps 500000 --micro_batch_size 2 --gradient_accumulation_steps 4 --max_length 4096 --eval_interval 2000 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name torchrun --initializer_range 0.02 --optimizer_eps 1e-8 --job_id 55874770 --node_size ${node_size} --line_index e_line/ --sampling random --resume "
#special_args=" --model_name micro --dtype float32 --total_steps 500000 --micro_batch_size 1 --gradient_accumulation_steps 4 --max_length 4096 --eval_interval 2000 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name torchrun --initializer_range 0.02 --optimizer_eps 1e-8 --job_id 55941537 --node_size ${node_size} --line_index e_line/ --sampling random --resume"
#special_args=" --model_name tiny --dtype float32 --total_steps 500000 --micro_batch_size 1 --gradient_accumulation_steps 4 --max_length 4096 --eval_interval 2000 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name torchrun --initializer_range 0.02 --optimizer_eps 1e-8 --job_id 55940523 --node_size ${node_size} --line_index e_line/ --sampling random --resume "
#special_args=" --model_name small --dtype float32 --total_steps 500000 --micro_batch_size 1 --gradient_accumulation_steps 2 --max_length 4096 --eval_interval 2000 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name torchrun --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size} --line_index e_line/ --sampling random --resume "
#special_args=" --model_name medium --dtype float32 --total_steps 500000 --micro_batch_size 1 --gradient_accumulation_steps 2 --max_length 4096 --eval_interval 10 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name torchrun --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size} --line_index e_line/ --sampling random  --resume"



###### redpajama megatron-deepspeed launch redpajama_megatron_deepspeed_nscc.py
source ./scripts/pretrain_nscc.sh # get variable from pretrain_nscc.sh $TOKENIZER_PATH $GPT_ARGS $TENSORBOARD_ARGS $DS_ARGS
APP_common="python -m torch.distributed.run --nnodes=${node_size} --nproc_per_node=${GPUS_PER_NODE} --node_rank=${world_rank} --master_addr=${master_ip} --master_port=8822  pretrain/redpajama_megatron_deepspeed_nscc.py "
special_args=" --tensor-model-parallel-size $TP --pipeline-model-parallel-size $PP --data-path $DATASET --data-impl mmap --tokenizer-type Llama2Tokenizer --tokenizer-model $TOKENIZER_PATH $GPT_ARGS $TENSORBOARD_ARGS $DS_ARGS "



###### test script
#APP="python -m torch.distributed.run --nnodes=${node_size} --nproc_per_node=4 --node_rank=${world_rank} --master_addr=${master_ip} --master_port=8822 pretrain/redpajama_nscc.py --local_rank ${world_rank} --model_name nano --dtype float32 --total_steps 50 --micro_batch_size 2 --gradient_accumulation_steps 2 --max_length 4096 --eval_interval 10 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name red_pajama_test --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size}"
###### test deepspeed
#APP="python -m torch.distributed.run --nnodes=${node_size} --nproc_per_node=4 --node_rank=${world_rank} --master_addr=${master_ip} --master_port=8822 pretrain/openwebtext_deepspeed_nscc.py --model_name nano --dtype float32 --total_steps 20 --micro_batch_size 2  --max_length 2048 --eval_interval 5 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name deepspeed --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size} "
#APP="deepspeed --num_gpus=${uni_size} --num_node=${node_size} --hostfile hostfile/hostfile-dl-$SLURM_JOB_ID --master_addr=${master_ip} --master_port=8822 pretrain/redpajama_deepspeed_nscc.py --local_rank ${world_rank} --model_name nano --dtype float32 --total_steps 20 --micro_batch_size 2  --gradient_accumulation_steps 2 --max_length 2048 --eval_interval 2 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name deepspeed --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size} "
#APP="deepspeed --num_gpus=${uni_size} --num_node=${node_size} train.py "
#APP="python pretrain/openwebtext_deepspeed_nscc.py --model_name nano --dtype float32 --total_steps 100 --micro_batch_size 2 --gradient_accumulation_steps 2 --max_length 2028 --eval_interval 20 --use_flash_attn False --learning_rate 6e-4 --calculate_ortho --run_name deepspeed --initializer_range 0.02 --optimizer_eps 1e-8 --job_id ${job_id} --node_size ${node_size} --line_index b_line/"
#deepspeed --num_gpus=4 --num_node=1 pretrain/openwebtext_deepspeed.py


APP=${APP_common}${special_args}

#export UCX_NET_DEVICES=mlx5_0:1
#export UCX_IB_PCI_BW=mlx5_0:50Gbs
#echo numactl --cpunodebind=0 --membind=0 ${APP}
#numactl --cpunodebind=0 --membind=0 ${APP}

case ${lrank} in
[0])
  export HIP_VISIBLE_DEVICES=0,1,2,3
  export UCX_NET_DEVICES=mlx5_0:1
  export UCX_IB_PCI_BW=mlx5_0:100Gbs
  echo numactl --cpunodebind=0 --membind=0 ${APP}
  numactl --cpunodebind=0 --membind=0 ${APP}
  ;;
[1])
  export HIP_VISIBLE_DEVICES=0,1,2,3
  export UCX_NET_DEVICES=mlx5_1:1
  export UCX_IB_PCI_BW=mlx5_1:100Gbs
  echo numactl --cpunodebind=1 --membind=1 ${APP}
  numactl --cpunodebind=1 --membind=1 ${APP}
  ;;
[2])
  export HIP_VISIBLE_DEVICES=0,1,2,3
  export UCX_NET_DEVICES=mlx5_2:1
  export UCX_IB_PCI_BW=mlx5_2:100Gbs
  echo numactl --cpunodebind=2 --membind=2 ${APP}
  numactl --cpunodebind=2 --membind=2 ${APP}
  ;;
[3])
  export HIP_VISIBLE_DEVICES=0,1,2,3
  export UCX_NET_DEVICES=mlx5_3:1
  export UCX_IB_PCI_BW=mlx5_3:100Gbs
  echo numactl --cpunodebind=3 --membind=3 ${APP}
  numactl --cpunodebind=3 --membind=3 ${APP}
  ;;
esac


#//单机四卡
#mpirun -np 1 `pwd`/single_torchrun.sh
