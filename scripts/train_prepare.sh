#!/bin/bash

a_line_global_batch=240
b_line_global_batch=480
c_line_global_batch=960

node_size=480
TP=4
PP=2
MICRO_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
data_parallel_size=`expr $node_size \* 4 / $PP / $TP`
GLOBAL_BATCH_SIZE=`expr $data_parallel_size \* $MICRO_BATCH_SIZE \* $GRADIENT_ACCUMULATION_STEPS`

HIDDEN_SIZE=2048 # e.g. llama-13b: 5120
FFN_HIDDEN_SIZE=$(echo "scale=0; $(echo "scale=3; $HIDDEN_SIZE*8/3/256+0.999" | bc)/1*256" | bc)
NUM_LAYERS=24 # e.g. llama-13b: 40
NUM_HEADS=32 # e.g. llama-13b: 40
SEQ_LENGTH=2048
NUM_KV_HEADS=8 # llama2 70B uses GQA

#CHECKPOINT_JOB_ID=random
CHECKPOINT_JOB_ID=52073265
ZERO_STAGE=0
MODEL_NAME='xl'
TRAIN_STEPS=700000
LR_WARMUP_STEPS=2000
LR=6e-4
MIN_LR=6e-5
fp16_flag=false

sed -i "s/#SBATCH -N .*/#SBATCH -N ${node_size}/"  ./multi_process.sh
sed -i "s/\bTP=.*/TP=${TP}/"  ./scripts/pretrain_nscc.sh
sed -i "s/\bPP=.*/PP=${PP}/"  ./scripts/pretrain_nscc.sh

sed -i "s/\bMICRO_BATCH_SIZE=.*/MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}/"  ./scripts/pretrain_nscc.sh
sed -i "s/\bGRADIENT_ACCUMULATION_STEPS=.*/GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}/"  ./scripts/pretrain_nscc.sh
sed -i "s/\bGLOBAL_BATCH_SIZE=.*/GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}/"  ./scripts/pretrain_nscc.sh

sed -i "s/\bHIDDEN_SIZE=.*/HIDDEN_SIZE=${HIDDEN_SIZE}/"  ./scripts/pretrain_nscc.sh
sed -i "s/\bFFN_HIDDEN_SIZE=.*/FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE}/"  ./scripts/pretrain_nscc.sh
sed -i "s/\bNUM_LAYERS=.*/NUM_LAYERS=${NUM_LAYERS}/"  ./scripts/pretrain_nscc.sh
sed -i "s/\bNUM_HEADS=.*/NUM_HEADS=${NUM_HEADS}/"  ./scripts/pretrain_nscc.sh
sed -i "s/\bSEQ_LENGTH=.*/SEQ_LENGTH=${SEQ_LENGTH}/"  ./scripts/pretrain_nscc.sh
sed -i "s/\bNUM_KV_HEADS=.*/NUM_KV_HEADS=${NUM_KV_HEADS}/"  ./scripts/pretrain_nscc.sh

sed -i "s/\bCHECKPOINT_JOB_ID=.*/CHECKPOINT_JOB_ID=${CHECKPOINT_JOB_ID}/"  ./scripts/pretrain_nscc.sh
sed -i "s/\bZERO_STAGE=.*/ZERO_STAGE=${ZERO_STAGE}/"  ./scripts/pretrain_nscc.sh
sed -i "s/\bMODEL_NAME=.*/MODEL_NAME=${MODEL_NAME}/"  ./scripts/pretrain_nscc.sh
sed -i "s/\bTRAIN_STEPS=.*/TRAIN_STEPS=${TRAIN_STEPS}/"  ./scripts/pretrain_nscc.sh
sed -i "s/\bLR_WARMUP_STEPS=.*/LR_WARMUP_STEPS=${LR_WARMUP_STEPS}/"  ./scripts/pretrain_nscc.sh
sed -i "s/\bLR=.*/LR=${LR}/"  ./scripts/pretrain_nscc.sh
sed -i "s/\bMIN_LR=.*/MIN_LR=${MIN_LR}/"  ./scripts/pretrain_nscc.sh

if  [ $fp16_flag == true ]
then
  sed -i "s/--no-gradient-accumulation-fusion.*/--no-gradient-accumulation-fusion --fp16 /"  ./scripts/pretrain_nscc.sh
  echo 'fp16'
else
  sed -i "s/--no-gradient-accumulation-fusion.*/--no-gradient-accumulation-fusion /"  ./scripts/pretrain_nscc.sh
  echo 'fp32'
fi

echo "TP is " $TP
echo "PP is " $PP
echo "node_size is " $node_size
echo "MICRO_BATCH_SIZE is " $MICRO_BATCH_SIZE
echo "data_parallel_size is " $data_parallel_size
echo "GRADIENT_ACCUMULATION_STEPS is " $GRADIENT_ACCUMULATION_STEPS
echo "GLOBAL_BATCH_SIZE is " $GLOBAL_BATCH_SIZE

echo "HIDDEN_SIZE is " $HIDDEN_SIZE
echo "FFN_HIDDEN_SIZE is " $FFN_HIDDEN_SIZE
echo "NUM_LAYERS is " $NUM_LAYERS
echo "NUM_HEADS is " $NUM_HEADS
echo "SEQ_LENGTH is " $SEQ_LENGTH
echo "NUM_KV_HEADS is " $NUM_KV_HEADS

echo "CHECKPOINT_JOB_ID is " $CHECKPOINT_JOB_ID
echo "ZERO_STAGE is " $ZERO_STAGE
echo "TRAIN_STEPS is " $TRAIN_STEPS
echo "LR_WARMUP_STEPS is " $LR_WARMUP_STEPS
echo "LR is " $LR
echo "MIN_LR is " $MIN_LR

#"gradient_accumulation_steps": $GRADIENT_ACCUMULATION_STEPS,
BASE_PATH=config
DS_CONFIG=${BASE_PATH}/deepspeed_${MODEL_NAME}.json
echo "DS_CONFIG name is: " $DS_CONFIG
rm -r ./$DS_CONFIG
cat <<EOT > $DS_CONFIG
  {
    "train_micro_batch_size_per_gpu" : $MICRO_BATCH_SIZE,
    "train_batch_size": $GLOBAL_BATCH_SIZE,
    "steps_per_print": 10,
    "zero_optimization": {
      "stage": $ZERO_STAGE
    },
    "fp16": {
      "enabled": $fp16_flag
    }
  }
EOT



