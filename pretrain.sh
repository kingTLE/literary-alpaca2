output_model=/kaggle/working/LiteraryAlpaca2
model=/kaggle/working
#定义输出模型的保存路径
if [ ! -d ${output_model} ];then
    mkdir ${output_model}
fi

#使用deepspeed启动分布式训练,指定使用2个GPU
deepspeed --num_gpus 2 /kaggle/input/pretrain-code/pretrain.py \
    --train_files /kaggle/input/random-data-1/Pre_training_random_data/*/*/*/* \
    --deepspeed /kaggle/input/pretrain-code/deepspeed_config.json \
    --output_dir ${output_model} \
    --use_fast_tokenizer false \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-4 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1 \
    --warmup_steps 1000 \
    --logging_dir ${model}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
    --logging_first_step True \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 2000 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 256 \
    --report_to tensorboard \
    --run_name ${output_model} \
    --gradient_checkpointing \
    --ignore_data_skip true \
    --ddp_timeout 18000000 \
    --load_in_kbits 4 \
    --torch_dtype auto \
    --lr_scheduler_type cosine \
    | tee -a ${model}/train.log
#    --resume_from_checkpoint ${output_model}/checkpoint-20400 \


