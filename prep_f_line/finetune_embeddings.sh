
RUN_NAME="finetune_embeddings"
LANGUAGE_MODEL="/data2/liuyang/llm_pretrain_xmodel/baseline_v8_64000"
DATA_PATH="/data1/wangqun/redpajama-v2-process/bins_xmodel_v8_64000_test4_unigram"
OUTPUT_DIR="./out/${RUN_NAME}/"

CUDA_VISIBLE_DEVICES="$1" torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc-per-node=8 prep_f_line/train_f_line_cc_embeddings.py \
    --model_name_or_path ${LANGUAGE_MODEL} \
    --data_path ${DATA_PATH} \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --max_steps 50000 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'DecoderLayer' \
    --tf32 True \
    --report_to tensorboard \
    --logging_dir "./runs/${RUN_NAME}/"
