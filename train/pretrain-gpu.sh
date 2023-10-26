output_model=/kaggle/working/LiteraryAlpaca2
model=/kaggle/working
dataset=/kaggle/working/dataset_cache
#dataset=/kaggle/input/cache1/dataset_cache
if [ ! -d ${output_model} ];then
    mkdir ${output_model}
fi
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
#                  /kaggle/input/random-data-1/Pre_training_random_data/*/*/*.txt \
#/kaggle/input/random-data-1/Pre_training_random_data/*/*/*/* \
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
torchrun --nnodes 1 --nproc_per_node 2 /kaggle/input/pretrain-code/pretrain.py \
    --train_files /kaggle/input/random-data-1/Pre_training_random_data/\(2\)/灵异/推理侦探/灵魂拼图作者伯百川.txt \
    --output_dir ${output_model} \
    --per_device_train_batch_size 1 \
    --preprocessing_num_workers 8 \
    --hub_token "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" \
    --push_to_hub true \
    --hub_model_id "taotie1/literary-alpaca2" \
    --learning_rate 1e-5 \
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
    --block_size 512 \
    --report_to tensorboard \
    --run_name ${output_model} \
    --ignore_data_skip true \
    --ddp_timeout 18000000 \
    --load_in_kbits 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --fp16 \
    | tee -a ${model}/train.log
#    --resume_from_checkpoint ${output_model}/checkpoint-20400 \

