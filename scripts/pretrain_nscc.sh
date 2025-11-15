#!/bin/bash
# This example script is contributed by external user https://github.com/nrailgun
#set -ex

######################################
# Change the below configurations here
comm_world_rank=$OMPI_COMM_WORLD_RANK
#DATASET_1="/data2/liuyang/openwebtext/openwebtext_text_document"
#DATASET_1="/public/home/chenhao/llm_test/llm_pretrain_xmodel/data/megatron_data/megatron_text_document"
#DATASET="/public/home/chenhao/llm_test/llm_pretrain_xmodel/data/openwebtext/openwebtext_text_document"

#DATASET_0="2.5 /public/home/chenhao/llm_test/llm_pretrain_xmodel/data/redpajama-data-1T/red_pajama_origin/arxiv_"
#DATASET_1="4.5 /public/home/chenhao/llm_test/llm_pretrain_xmodel/data/redpajama-data-1T/red_pajama_origin/book_"
#DATASET_2="15 /public/home/chenhao/llm_test/llm_pretrain_xmodel/data/redpajama-data-1T/red_pajama_origin/c4_"
#DATASET_3="67 /public/home/chenhao/llm_test/llm_pretrain_xmodel/data/redpajama-data-1T/red_pajama_origin/cc20"
#DATASET_4="4.5 /public/home/chenhao/llm_test/llm_pretrain_xmodel/data/redpajama-data-1T/red_pajama_origin/github_"
#DATASET_5="2.0 /public/home/chenhao/llm_test/llm_pretrain_xmodel/data/redpajama-data-1T/red_pajama_origin/stackexchange_"
#DATASET_6="4.5 /public/home/chenhao/llm_test/llm_pretrain_xmodel/data/redpajama-data-1T/red_pajama_origin/wikipedia_"
#DATASET="${DATASET_0} ${DATASET_1} ${DATASET_2} ${DATASET_3} ${DATASET_4} ${DATASET_5} ${DATASET_6}"


DATASET_0="0.0160 /public/home/chenhao/llm_test/llm_pretrain_xmodel/data/data_e_line/origin_data/arxiv"
DATASET_1="0.0300 /public/home/chenhao/llm_test/llm_pretrain_xmodel/data/data_e_line/origin_data/book"
DATASET_2="0.1000 /public/home/chenhao/llm_test/llm_pretrain_xmodel/data/data_e_line/origin_data/c4"
DATASET_3="0.5451 /public/home/chenhao/llm_test/llm_pretrain_xmodel/data/data_e_line/origin_data/cc"
DATASET_4="0.0150 /public/home/chenhao/llm_test/llm_pretrain_xmodel/data/data_e_line/origin_data/github"
DATASET_5="0.0536 /public/home/chenhao/llm_test/llm_pretrain_xmodel/data/data_e_line/origin_data/starcoder"
DATASET_6="0.0174 /public/home/chenhao/llm_test/llm_pretrain_xmodel/data/data_e_line/origin_data/stackexchange"
DATASET_7="0.0520 /public/home/chenhao/llm_test/llm_pretrain_xmodel/data/data_e_line/origin_data/wiki"
DATASET_8="0.0003 /public/home/chenhao/llm_test/llm_pretrain_xmodel/data/data_e_line/origin_data/amps"
DATASET_9="0.0006 /public/home/chenhao/llm_test/llm_pretrain_xmodel/data/data_e_line/origin_data/Bookcorpus"
DATASET_10="0.0020 /public/home/chenhao/llm_test/llm_pretrain_xmodel/data/data_e_line/origin_data/fanfics"
DATASET_11="0.0080 /public/home/chenhao/llm_test/llm_pretrain_xmodel/data/data_e_line/origin_data/openwebmath"
DATASET_12="0.0100 /public/home/chenhao/llm_test/llm_pretrain_xmodel/data/data_e_line/origin_data/pubmed"
DATASET_13="0.0100 /public/home/chenhao/llm_test/llm_pretrain_xmodel/data/data_e_line/origin_data/law"
DATASET_14="0.0050 /public/home/chenhao/llm_test/llm_pretrain_xmodel/data/data_e_line/origin_data/news"
DATASET_15="0.0050 /public/home/chenhao/llm_test/llm_pretrain_xmodel/data/data_e_line/origin_data/patent"
DATASET_16="0.0644 /public/home/chenhao/llm_test/llm_pretrain_xmodel/data/data_e_line/origin_data/webtext"
DATASET_17="0.0656 /public/home/chenhao/llm_test/llm_pretrain_xmodel/data/data_e_line/origin_data/PTD"
DATASET="${DATASET_0} ${DATASET_1} ${DATASET_2} ${DATASET_3} ${DATASET_4} ${DATASET_5} ${DATASET_6} ${DATASET_7} ${DATASET_8} ${DATASET_9} ${DATASET_10} ${DATASET_11} ${DATASET_12} ${DATASET_13} ${DATASET_14} ${DATASET_15} ${DATASET_16} ${DATASET_17}"


JOB_ID=$SLURM_JOB_ID
CHECKPOINT_JOB_ID=52073265
if [ $CHECKPOINT_JOB_ID == random ]
then
  TENSORBOARD_JOB_ID=$JOB_ID
  echo 'tensorboard_job_id is ' $TENSORBOARD_JOB_ID
else
  TENSORBOARD_JOB_ID=$CHECKPOINT_JOB_ID
  JOB_ID=$CHECKPOINT_JOB_ID
  echo 'tensorboard_job_id is ' $TENSORBOARD_JOB_ID
  echo 'job_id is ' $JOB_ID
fi
CHECKPOINT_PATH=./out/megatron_model/model_$CHECKPOINT_JOB_ID
SAVE_PATH=./out/megatron_model/model_$JOB_ID
#TOKENIZER_PATH=/public/home/chenhao/llm_test/llm_pretrain_xmodel/tokenizers/llama/tokenizer.model # offical llama tokenizer.model
TOKENIZER_PATH=/public/home/chenhao/llm_test/llm_pretrain_xmodel/tokenizers/xmodel/v6/xmodel_32000/xmodel_32000.model

TP=1
PP=4
ZERO_STAGE=0

GPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=8822
NNODES=$OMPI_UNIVERSE_SIZE
#NODE_RANK=0

HIDDEN_SIZE=2048 # e.g. llama-13b: 5120
FFN_HIDDEN_SIZE=$(echo "scale=0; $(echo "scale=3; $HIDDEN_SIZE*8/3/256+0.999" | bc)/1*256" | bc)   # e.g. llama-13b: 13824
NUM_LAYERS=24 # e.g. llama-13b: 40
NUM_HEADS=16 # e.g. llama-13b: 40
SEQ_LENGTH=2048
NUM_KV_HEADS=4 # llama2 70B uses GQA

# e.g. llama: 4M tokens
TRAIN_STEPS=10000 # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps
MODEL_NAME='xl'
SAVE_INTERVAL=2000
EVAL_INTERVAL=5000

LR=2e-4
MIN_LR=2e-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

uni_size=$OMPI_UNIVERSE_SIZE # =gpu num
echo " uni_size is " $uni_size
MICRO_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
data_parallel_size=`expr $uni_size / $PP / $TP`
GLOBAL_BATCH_SIZE=1
echo "FFN_HIDDEN_SIZE is " $FFN_HIDDEN_SIZE
echo "MICRO_BATCH_SIZE is " $MICRO_BATCH_SIZE
echo "data_parallel_size is " $data_parallel_size
echo "GLOBAL_BATCH_SIZE is " $GLOBAL_BATCH_SIZE

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

BASE_PATH=./config
DS_CONFIG=${BASE_PATH}/deepspeed_${MODEL_NAME}.json
echo "DS_CONFIG name is: " $DS_CONFIG

DS_ARGS=""
DS_ARGS=" --deepspeed ${DS_ARGS}"
DS_ARGS=" --deepspeed_config=$DS_CONFIG ${DS_ARGS}"
DS_ARGS=" --zero-stage=$ZERO_STAGE ${DS_ARGS}"

if [ "${activation_checkpoint}" = "true" ]; then
  DS_ARGS="--deepspeed-activation-checkpointing ${DS_ARGS}"

  ## old argument for recomputing the transformer layer
  # DS_ARGS="--checkpoint-activations ${DS_ARGS}"

  ## new argument for recomputing the transformer layer
#  DS_ARGS="--recompute-granularity full --recompute-method uniform ${DS_ARGS}"
  ## new argument for recomputing only the attention layer
   DS_ARGS="--recompute-granularity selective ${DS_ARGS}"
fi


#DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

GPT_ARGS="--num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --ffn-hidden-size $FFN_HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --train-iters $TRAIN_STEPS \
       --save $SAVE_PATH \
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
       --save-interval $SAVE_INTERVAL \
       --eval-interval $EVAL_INTERVAL \
       --eval-iters 10 \
       --no-query-key-layer-scaling \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --swiglu \
       --normalization rmsnorm \
       --disable-bias-linear \
       --num-key-value-heads $NUM_KV_HEADS \
       --train-doc-idx-path fffd35e023f79f4ac399dac81898e9a8_doc_idx.npy \
       --train-desc-path fffd35e023f79f4ac399dac81898e9a8.dsc \
       --train-sample-idx-path fffd35e023f79f4ac399dac81898e9a8_sample_idx.npy \
       --train-shuffle-idx-path fffd35e023f79f4ac399dac81898e9a8_shuffle_idx.npy \
       --seed 1235 \
       --no-gradient-accumulation-fusion \

       "

TENSORBOARD_ARGS=" \
    --tensorboard-dir runs/e_line/${MODEL_NAME}_openwebtext_megatron_${TENSORBOARD_JOB_ID} \
    --tensorboard-queue-size 5 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --log-memory-to-tensorboard \
    "

#torchrun $DISTRIBUTED_ARGS \
#       pretrain/redpajama_megatron_deepspeed_nscc.py \
#       --tensor-model-parallel-size $TP \
#       --pipeline-model-parallel-size $PP \
#       --data-path $DATASET \
#       --data-impl mmap \
#       --tokenizer-type Llama2Tokenizer \
#       --tokenizer-model $TOKENIZER_PATH \
#       $GPT_ARGS \
#       $TENSORBOARD_ARGS \
#       $DS_ARGS
