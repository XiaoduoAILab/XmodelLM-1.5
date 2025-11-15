#!/bin/bash

#SBATCH -p normal
#SBATCH --exclusive
#SBATCH --exclude=d13r3n14,a08r2n09,b06r3n03,a09r1n06,a09r1n07,a09r1n00,d09r1n11,e14r1n19,a08r2n16,a10r3n00,a09r1n10,d16r2n12,d09r1n03,e13r2n12,b16r3n03,b16r3n04,b06r1n13,b09r2n11,b10r3n10,b06r1n00,c01r2n12,b09r4n01,e16r3n00,b07r2n10,e15r1n18,b05r3n11,b05r4n15,b11r4n00,b11r2n15,e15r3n13,c01r4n04,b06r2n00,b06r2n09,b06r3n10,a02r3n03,a01r2n00,a09r2n15,a09r2n15,b06r4n07,a08r2n06,a13r3n18,a01r2n03,b07r4n07,a06r4n19,e15r3n14
#SBATCH -N 30
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=dcu:4
#SBATCH -J GPT2
#SBATCH -o ./nscc_log/GPT2-%j.out
#SBATCH -e ./nscc_log/error_log-%j.out

rm -rf ./hostfile/*
source prepare_env.sh
echo "env prepare: $(date)"


#cd /public/home/chenhao/llm_test/llm_pretrain_xmodel/
# set master node ip
echo "master node start multi process"
if [ ! -f "master_node_ip.txt" ];then
  echo "master node ip file is empty"
  touch master_node_ip.txt
  else
    rm master_node_ip.txt
fi


echo "START TIME: $(date)"
hostfile=./hostfile/$SLURM_JOB_ID
scontrol show hostnames $SLURM_JOB_NODELIST > ${hostfile}
for i in `cat $hostfile`
do
    echo ${i} slots=4 >> `pwd`/hostfile/hostfile-dl-$SLURM_JOB_ID
done
np=$(cat $hostfile|sort|uniq |wc -l)
echo "number of node :" $np
node_name_list=$(cat $hostfile |sed -n "1p")
#echo "node_name_list is: " ${node_name_list}
dist_url=`echo $node_name_list | awk '{print $1}'`

# 节点列表
nodes_list=$(scontrol show hostname $SLURM_JOB_NODELIST)
# 对第一个节点赋值为主节点
master_node=$(head -n 1 <<< "$nodes_list")
master_node_ip=$(srun --nodes=1 --ntasks=1 -w "$master_node" hostname --ip-address)
echo "master node ip is : $master_node_ip"
# 计数器
node_count=0
# 一共的节点数
node_num=($(echo $nodes_list | tr " " "\n" | wc -l))
echo "node_num: " $node_num

mas_ip=`ifconfig |grep inet | grep 10. |grep -v 127.0.0.1|grep -v inet6 |awk '{print $2}'|tr -d 'addr:'`

echo "check again master node ip is : $mas_ip"
echo "$master_node_ip" > master_node_ip.txt

# 打印 master_node
echo "slurm_node_id: " $SLURM_NODEID
echo "nodes_list is: " ${nodes_list}
echo "master_node: " $master_node
master_node_ip=$(srun --nodes=1 --ntasks=1 -w "$$master_node" hostname --ip-address)
echo "master_node_ip: " $master_node_ip

ip_list=()
for cur_node in $nodes_list; do
    echo "node is: " ${cur_node}
    head_node=${cur_node}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
    echo "head_node_ip is: " ${head_node_ip}
    ip_list+=(${head_node_ip})
done
wait
echo "ip_list_output" ${ip_list[@]} ${nodes_list}


dp_np=$(($np*4))
echo "dp_np is " $dp_np
#new_np=`expr $np - 1`
#torchrun
mpirun --allow-run-as-root -np $np --map-by node:pe=1 --hostfile hostfile/hostfile-dl-$SLURM_JOB_ID `pwd`/single_process.sh
#deepspeed
#mpirun --allow-run-as-root -np $np --map-by node:pe=1 --hostfile hostfile/hostfile-dl-$SLURM_JOB_ID `pwd`/single_process.sh
#mpirun --allow-run-as-root -np $dp_np --hostfile hostfile/hostfile-dl-$SLURM_JOB_ID  --bind-to none `pwd`/single_process.sh dist_url
