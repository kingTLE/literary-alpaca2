output_model=/kaggle/working/LiteraryAlpaca2
model=/kaggle/working
dataset=/kaggle/working/dataset_cache
#dataset=/kaggle/input/dataset_cache
export XLA_USE_BF16=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
python  /kaggle/input/pretrain-code/pretrain-tpu.py \
    --train_files /kaggle/input/random-data-1/Pre_training_random_data/*/*/*/* \
                  /kaggle/input/random-data-1/Pre_training_random_data/*/*/*.txt \
    --per_device_train_batch_size 3 \
    --preprocessing_num_workers 20 \
    --run_name ${output_model} \
    --output_dir ${output_model} \
    --hub_token "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" \
    --push_to_hub true \
    --hub_model_id "taotie1/literary-alpaca2-7B" \
    --use_fast_tokenizer false\
    --learning_rate 3e-5 \
    --data_cache_dir ${dataset} \
    --num_train_epochs 3 \
    --warmup_steps 1000 \
    --logging_dir ${model}/logs \
    --logging_strategy steps \
    --logging_steps 200 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 2000 \
    --seed 42 \
    --bf16 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 4096 \
    --report_to tensorboard \
    --ignore_data_skip true \
    --ddp_timeout 18000000 \
    | tee -a ${model}/trains.log
#    --resume_from_checkpoint ${output_model}/checkpoint-20400 \
#tensorboard --logdir logs


