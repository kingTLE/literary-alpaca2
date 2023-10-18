output_model=/kaggle/working/LiteraryAlpaca2
model=/kaggle/working

python  /kaggle/input/pretrain-code/pretrain-tpu.py \
    --train_files /kaggle/input/random-data-1/Pre_training_random_data/*/*/*/* \
                  /kaggle/input/random-data-1/Pre_training_random_data/*/*/*.txt \
    --per_device_train_batch_size 8 \
    --preprocessing_num_workers 20 \
    --output_dir ${output_model} \
    --use_fast_tokenizer false\
    --learning_rate 3e-5 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --warmup_steps 1000 \
    --logging_dir ${model} \
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
    --gradient_checkpointing \
    --ignore_data_skip true \
    --ddp_timeout 18000000 \
    | tee -a ${model}/trains.log
#    --resume_from_checkpoint ${output_model}/checkpoint-20400 \
#tensorboard --logdir logs


