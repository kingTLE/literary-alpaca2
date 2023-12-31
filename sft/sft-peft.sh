output_model=/root/autodl-tmp/sft-lora
model=/root/autodl-fs/train2/working

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
torchrun --nnodes 1 --nproc_per_node 1 /root/autodl-fs/train/sft-peft.py \
    --deepspeed /root/autodl-fs/train/deepspeed_config_sft.json \
    --data_cache_dir /root/autodl-tmp/data \
    --train_files /root/autodl-fs/train/data  \
    --output_dir ${output_model} \
    --hub_token "" \
    --push_to_hub false \
    --hub_model_id "" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --preprocessing_num_workers 15 \
    --load_best_model_at_end  True \
    --logging_first_step True \
    --lr_scheduler_type cosine \
    --learning_rate 1e-4 \
    --use_fast_tokenizer false\
    --num_train_epochs 1 \
    --logging_dir '/root/tf-logs' \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_steps 1000 \
    --eval_steps 250  \
    --save_total_limit 5 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 4096 \
    --report_to tensorboard \
    --ignore_data_skip true \
    --ddp_timeout 18000000 \
    --load_in_kbits 4 \
    --gradient_checkpointing \
    --bf16 \
    --gradient_accumulation_steps 16 \
    | tee -a ${model}/trains.log

#    --resume_from_checkpoint ${output_model}/checkpoint-20400 \
    # --peft_path /root/autodl-tmp/lora \
