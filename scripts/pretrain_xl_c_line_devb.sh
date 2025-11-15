#!/bin/bash
# This example script is contributed by external user https://github.com/nrailgun
set -ex

######################################
# Change the below configurations here
BASE_PATH=./tmp
DS_CONFIG=${BASE_PATH}/deepspeed.json
#DATASET_1="./tmp/data/bookcorpus_train_1m_text_sentence"
#DATASET_1="/data2/liuyang/lit-redpajama-new/"
#DATASET="1 ${DATASET_1}"
DATASET="/data2/liuyang/lit-redpajama-new/"
CHECKPOINT_PATH=./out/xl_megatron
TOKENIZER_PATH=tokenizers/llama/tokenizer.model # official llama tokenizer.model

TP=1
PP=2
ZERO_STAGE=0

GPUS_PER_NODE=2
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0

HIDDEN_SIZE=2048 # e.g. xmodel_32000-xl: 2048
FFN_HIDDEN_SIZE=5632 # e.g. xmodel_32000-xl: 5632
NUM_LAYERS=24 # e.g. xmodel_32000-xl: 24
NUM_HEADS=32 # e.g. xmodel_32000-xl: 32
SEQ_LENGTH=4096
NUM_KV_HEADS=8 # xmodel_32000-xl uses GQA

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=960 # e.g. c_line: 4M tokens
TRAIN_STEPS=350000 # e.g. c_line: 1.4T tokens / 4M tokens_per_batch = 350000 steps
LR=6e-4
MIN_LR=6e-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

## Activation checkpointing saves GPU memory, but reduces training speed
# activation_checkpoint="true"
activation_checkpoint="false"

# Below configuration required for llama model as per llama paper
# --no-query-key-layer-scaling \
# --attention-dropout 0 \
# --hidden-dropout 0 \
# --use-rotary-position-embeddings \
# --untie-embeddings-and-output-weights \
# --swiglu \
# --normalization rmsnorm \
# --disable-bias-linear \
######################################

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 10,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": true
  }
}
EOT

DS_ARGS=""
DS_ARGS=" --deepspeed ${DS_ARGS}"
DS_ARGS=" --deepspeed_config=$DS_CONFIG ${DS_ARGS}"
DS_ARGS=" --zero-stage=$ZERO_STAGE ${DS_ARGS}"

if [ "${activation_checkpoint}" = "true" ]; then
  DS_ARGS="--deepspeed-activation-checkpointing ${DS_ARGS}"

  ## old argument for recomputing the transformer layer
  # DS_ARGS="--checkpoint-activations ${DS_ARGS}"

  ## new argument for recomputing the transformer layer
  DS_ARGS="--recompute-granularity full --recompute-method uniform ${DS_ARGS}"
  ## new argument for recomputing only the attention layer
  # DS_ARGS="--recompute-granularity selective ${DS_ARGS}"
fi


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

GPT_ARGS="--num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --ffn-hidden-size $FFN_HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --train-iters $TRAIN_STEPS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr $LR \
       --lr-decay-style cosine \
       --min-lr $MIN_LR \
       --weight-decay $WEIGHT_DECAY \
       --clip-grad $GRAD_CLIP \
       --lr-warmup-iters $LR_WARMUP_STEPS \
       --optimizer adam \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 10 \
       --save-interval 2000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --bf16 \
       --use-flash-attn-v2 \
       --no-query-key-layer-scaling \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --swiglu \
       --normalization rmsnorm \
       --disable-bias-linear \
       --num-key-value-heads $NUM_KV_HEADS \
       "

TENSORBOARD_ARGS=" \
    --tensorboard-dir runs/large_openwebtext_megatron \
    --tensorboard-queue-size 5 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --log-memory-to-tensorboard \
    "

torchrun $DISTRIBUTED_ARGS \
       pretrain/pretrain_c_line.py \
       --tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --data-path $DATASET \
       --data-impl mmap \
       --tokenizer-type Llama2Tokenizer \
       --tokenizer-model $TOKENIZER_PATH \
       $GPT_ARGS \
       $TENSORBOARD_ARGS \
       $DS_ARGS
