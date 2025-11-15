#!/bin/bash
# megatron to transformers: You need to copy the tokenizer files into the save_path
# bash model_convertor.sh ../../Megatron-LM/ ../../llama-hf2mg-test-2-2/release/ ../../llama_mg2hf 1 1 llama-7b 1 true
# transformers to megatron
# bash model_convertor.sh ../../Megatron-LM/ ../../llama-7b-hf ../../llama-hf2mg 1 1 llama-7b 1 false
set -e
START_TIME=$SECONDS

root_path=/data/huangxucheng/llm_pretrain_xmodel
#root_path=/public/home/chenhao/llm_test/llm_pretrain_xmodel

MEGATRON_PATH=root_path
SOURCE_CKPT_PATH=${root_path}/out/large_megatron/global_step100
TARGET_CKPT_PATH=${root_path}/out/large_megatron/torch_bin_dir
TP=2
PP=2
MN=llama-1b #llama-7b, llama-13b, llama-30b, llama-65b, llama2-7b, llama2-13b, llama2-70b
EXTRA_VOCAB_SIZE=0
mg2hf=true

if [ $mg2hf = true ]; then
    do_options="
                --convert_checkpoint_from_megatron_to_transformers
    "
elif [ $mg2hf = false ]; then
    do_options=""
fi

export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH

python checkpoint_reshaping_and_interoperability.py \
--convert_checkpoint_from_megatron_to_transformers \
--load_path ${SOURCE_CKPT_PATH} \
--save_path ${TARGET_CKPT_PATH} \
--target_params_dtype fp32 \
--megatron_path ${MEGATRON_PATH} \
--target_tensor_model_parallel_size ${TP} \
--target_pipeline_model_parallel_size ${PP} \
--model_name ${MN} \
--extra_num_vocabs ${EXTRA_VOCAB_SIZE} \
${do_options}

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
