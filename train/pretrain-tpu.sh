output_model=/kaggle/working/LiteraryAlpaca2
model=/kaggle/working
dataset=/kaggle/working/dataset_cache
#dataset=/kaggle/input/cache2/dataset_cache
python  /kaggle/input/pretrain-code/pretrain-tpu.py \
    --train_files \
    --preprocessing_num_workers 30 \
    --output_dir ${output_model} \
    --hub_token  \
    --hub_model_id  \
    --use_fast_tokenizer false\
    --learning_rate 1e-5 \
    --data_cache_dir ${dataset} \
    --block_size 128 \
    --report_to tensorboard \
    | tee -a ${model}/trains.log
#    --resume_from_checkpoint ${output_model}/checkpoint-20400 \
#tensorboard --logdir logs
#                  /kaggle/input/random-data-1/Pre_training_random_data/*/*/*.txt \


