output_model=/root/autodl-tmp/LiteraryAlpaca2
model=/root/autodl-fs/train/working
dataset=/root/autodl-fs/train/working/dataset_cache
#dataset=/root/train/working/dataset_cache/data

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

torchrun --nnodes 1 --nproc_per_node 1 /root/autodl-fs/train/pretrain-peft1.py \
    --deepspeed /root/autodl-fs/train/deepspeed_config_peft1.json \
    --train_files /root/autodl-fs/train/random_data/Pre_training_random_data/*/*/*.txt \
                  /root/autodl-fs/train/random_data/Pre_training_random_data/*/*/*/*  \
    --output_dir ${output_model} \
    --per_device_train_batch_size 1 \
    --preprocessing_num_workers 15 \
    --hub_token "" \
    --push_to_hub false \
    --hub_model_id "taotie1/literary-alpaca2-13B" \
    --lr_scheduler_type cosine \
    --learning_rate 2e-5 \
    --use_fast_tokenizer false\
    --data_cache_dir ${dataset} \
    --num_train_epochs 1 \
    --logging_dir '/root/tf-logs' \
    --logging_strategy steps \
    --logging_steps 1 \
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
    --load_in_kbits 16 \
    --gradient_checkpointing \
    --fp16 \
    --gradient_accumulation_steps 8 \
    | tee -a ${model}/trains.log
#    --resume_from_checkpoint ${output_model}/checkpoint-20400 \

