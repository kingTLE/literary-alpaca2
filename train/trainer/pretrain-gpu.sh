output_model=/root/train/working/LiteraryAlpaca2
model=/root/train/working
dataset=/root/train/working/dataset_cache
#dataset=/root/train/working/dataset_cache/data

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
torchrun --nnodes 1 --nproc_per_node 1 /root/train/pretrain-gpu.py \
    --deepspeed /root/train/deepspeed_config.json \
    --train_files /root/autodl-tmp/random_data/Pre_training_random_data/*/*/*.txt \
                  /root/autodl-tmp/random_data/Pre_training_random_data/*/*/*/*  \
    --output_dir ${output_model} \
    --per_device_train_batch_size 2 \
    --preprocessing_num_workers 15 \
    --hub_token "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" \
    --push_to_hub true \
    --hub_model_id "taotie1/literary-alpaca2" \
    --learning_rate 2e-5 \
    --use_fast_tokenizer false\
    --data_cache_dir ${dataset} \
    --num_train_epochs 1 \
    --warmup_steps 1000 \
    --logging_dir ${model}/logs \
    --logging_strategy steps \
    --logging_steps 200 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 2000 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 4096 \
    --report_to tensorboard \
    --run_name ${output_model} \
    --ignore_data_skip true \
    --ddp_timeout 18000000 \
    --load_in_kbits 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --fp16 \
    | tee -a ${model}/trains.log
#    --resume_from_checkpoint ${output_model}/checkpoint-20400 \

