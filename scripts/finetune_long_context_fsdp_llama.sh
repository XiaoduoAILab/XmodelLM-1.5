
RUN_NAME="xmodel_long_context_$2_llama"
LANGUAGE_MODEL="/data2/liuyang/models/xmodel/llama/"
TOKENIZER_PATH="/data2/liuyang/llm_pretrain_xmodel/tokenizers/xmodel/v6/xmodel_32000/"
OUTPUT_DIR="./out/${RUN_NAME}/"

if [ -z "$3" ]
  then
    DATA_PATH="/data1/liuyang/e_line_v6_32000/"
  else
    DATA_PATH="$3"
fi

CUDA_VISIBLE_DEVICES="$1" torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc-per-node=8 prep_f_line/train_long_context_llama.py \
    --model_name_or_path ${LANGUAGE_MODEL} \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --data_path "${DATA_PATH}" \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --max_steps 1500 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 5 \
    --model_max_length 32768 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --report_to tensorboard \
    --logging_dir "./runs/${RUN_NAME}/" \
    --gradient_checkpointing 1
