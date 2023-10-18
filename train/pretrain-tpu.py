import logging
import os
import sys
import time
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List

import tensorflow as tf
import transformers
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    set_seed,
    TrainingArguments,
    HfArgumentParser,
    Trainer,
    default_data_collator,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState
)

logger = logging.getLogger(__name__)


# 添加 ModelArguments 数据类
@dataclass
class ModelArguments:
    pretrained_model_name: Optional[str] = field(default="/kaggle/input/llama-2/pytorch/13b-hf/1")
    tokenizer_name: Optional[str] = field(default="/kaggle/input/merged-tokenizer/incorporation_hf2")
    use_fast_tokenizer: bool = field(default=True)


# 添加 DataTrainingArguments 数据类
@dataclass
class DataTrainingArguments:
    train_files: Optional[List[str]] = field(default=None)
    keep_linebreaks: bool = field(default=True)
    block_size: Optional[int] = field(default=None)
    preprocessing_num_workers: Optional[int] = field(default=None)
    overwrite_cache: bool = field(default=False)


@dataclass
class MyTrainingArguments(TrainingArguments):
    max_steps: Optional[int] = field(default=None)
    use_cache: Optional[bool] = field(default=False)
    use_reentrant: Optional[bool] = field(default=False)

def check_and_create_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, trainer, early_stopping_patience, min_delta=0.001):
        self.trainer = trainer
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def on_evaluate(self, args, state, control, metrics):
        current_loss = metrics['loss']

        if self.best_loss - current_loss > self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.early_stopping_patience:
                control.should_training_stop = True  # 停止训练
                # 保存最佳模型
                if args.local_rank in [-1, 0]:
                    save_model_dir = args.output_dir
                    self.trainer.save_model(save_model_dir)
                    print(f"保存最佳模型到 {save_model_dir}")

    def on_train_end(self, args, state, control, **kwargs):
        if self.best_loss != float('inf'):
            print(f"早停：损失函数改进达到 {self.min_delta}，最佳损失函数值: {self.best_loss}")


class LossLoggingCallback(TrainerCallback):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        self.steps = 0

    def on_evaluate(self, args, state: TrainerState, control, metrics):
        loss = metrics.get("loss", None)
        if loss is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar("loss", loss, self.steps)
            self.steps += 1





def main():
    # 加载模型和数据
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"进程rank: {training_args.local_rank}，设备: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"分布式训练: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"训练参数 {training_args}")
    set_seed(training_args.seed)

    check_and_create_dir(training_args.output_dir)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = tf.train.latest_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"输出目录 ({training_args.output_dir}) 已存在且不为空"
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"检测到检查点，在{last_checkpoint}恢复训练"
            )

    # 加载分词器
    logger.info('开始加载分词器')
    tokenizer_kwargs = {
        "use_fast": model_args.use_fast_tokenizer,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    tokenizer.add_eos_token = True
    tokenizer.pad_token = tokenizer.eos_token
    logger.info('加载分词器结束')

    # 加载数据集
    data_files = {"train": data_args.train_files} if data_args.train_files is not None else {}
    if data_files["train"] is not None:
        extension = data_files["train"][0].split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args = {"keep_linebreaks": data_args.keep_linebreaks}
        else:
            dataset_args = {}

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=os.path.join(training_args.output_dir, 'dataset_cache'),
        **dataset_args,
    )
    # 数据处理和映射
    column_names = list(raw_datasets["train"].features)
    print("column_names为：", column_names)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        output = tokenizer(['<s>' + item + '</s>' for item in examples[text_column_name]])
        return output

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    block_size = min(data_args.block_size, tokenizer.model_max_length) if data_args.block_size else min(1024,
                                                                                                        tokenizer.model_max_length)

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        # concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        logger.info("组合文本输入示例长度%d，分组后大小%d" % (len(examples['input_ids']), len(result["input_ids"])))
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        batch_size=80000,
    )
    train_dataset = lm_datasets["train"]
    print("数据集长度：", len(train_dataset))

    config = AutoConfig.from_pretrained(model_args.pretrained_model_name)

    logger.info('开始加载模型')

    model = AutoModelForCausalLM.from_pretrained(
        model_args.pretrained_model_name,
        config=config,
        from_tf=bool(".ckpt" in model_args.pretrained_model_name),
    )
    tpu_devices = strategy.extended.worker_devices
    logger.info(f"模型加载在设备{tpu_devices}")

    for name, param in model.named_parameters():
        if "model.embed_tokens" not in name:
            param.requires_grad = False
    logger.info('加载模型结束')

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=16)

    training_args.fp16 = True

    def init_trainer(model, training_args, train_dataset, tokenizer, log_level):
        max_steps = (
            training_args.max_steps if training_args.max_steps is not None else int((len(
                train_dataset) * training_args.num_train_epochs) / (
                                                                                            training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps))
        )
        training_args.max_steps = max_steps
        logger.setLevel(logging.DEBUG)
        logger.info('开始初始化Trainer')
        start_time = time.time()
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f'结束初始化Trainer，用时 {elapsed_time:.2f} 秒')
        logger.setLevel(log_level)
        return trainer

    trainer = init_trainer(model, training_args, train_dataset, tokenizer, log_level)
    # 添加早停回调
    early_stopping_callback = EarlyStoppingCallback(trainer, early_stopping_patience=3)
    trainer.add_callback(early_stopping_callback)
    # 初始化 LossLoggingCallback
    loss_logging_callback = LossLoggingCallback("/kaggle/working/loss")
    trainer.add_callback(loss_logging_callback)

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    logger.info('开始训练')
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def get_strategy():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(resolver)

    tf.tpu.experimental.initialize_tpu_system(resolver)

    return tf.distribute.TPUStrategy(resolver)


if __name__ == "__main__":
    strategy = get_strategy()

    with strategy.scope():
        main()
