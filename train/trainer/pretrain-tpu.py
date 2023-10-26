import logging
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List

import datasets
import tensorflow as tf
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
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
)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    pretrained_model_name: Optional[str] = field(default="/kaggle/input/llama-2/pytorch/7b-hf/1")
    tokenizer_name: Optional[str] = field(default="/kaggle/input/merged-tokenizer/incorporation_hf2")
    use_fast_tokenizer: bool = field(default=True)


@dataclass
class DataTrainingArguments:
    train_files: Optional[List[str]] = field(default=None)
    keep_linebreaks: bool = field(default=True)
    block_size: Optional[int] = field(default=None)
    preprocessing_num_workers: Optional[int] = field(default=None)
    data_cache_dir: Optional[str] = field(default="/kaggle/working")


@dataclass
class MyTrainingArguments(TrainingArguments):
    max_steps: Optional[int] = field(default=None)
    use_cache: Optional[bool] = field(default=False)
    use_reentrant: Optional[bool] = field(default=False)
    # tpu_num_cores: Optional[int] = field(default=xm.xrt_world_size())
    tpu_num_cores: Optional[int] = field(default=8)
    output_dir: Optional[str] = field(default=tempfile.mkdtemp())
    run_name: Optional[str] = field(default=output_dir)


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


class MyTrainer(Trainer):
    def training_step(self, model, inputs):
        model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        print(self.optimizer)
        print(hasattr(self.optimizer, 'param_groups'))
        xm.optimizer_step(self.optimizer)
        xm.mark_step()

        return loss


def init_trainer(model, training_args, train_dataset, tokenizer, log_level):
    max_steps = (
        training_args.max_steps if training_args.max_steps is not None else int((len(
            train_dataset) * training_args.num_train_epochs) / (
                                                                                    training_args.per_device_train_batch_size))
    )
    training_args.max_steps = max_steps
    logger.setLevel(logging.DEBUG)
    logger.info('开始初始化Trainer')
    start_time = time.time()
    trainer = MyTrainer(
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


def check_and_create_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def preprocess_data(index,data_args, tokenizer):
    block_size = min(data_args.block_size, tokenizer.model_max_length) if data_args.block_size else min(1024,
                                                                                                        tokenizer.model_max_length)
    data_files = data_args.train_files

    num_files = len(data_files)
    num_files_per_proc = num_files // 8
    if index == 7:
        my_files = data_files[index * num_files_per_proc: num_files + 1]
    else:
        my_files = data_files[index * num_files_per_proc: (index + 1) * num_files_per_proc]

    file_or_names = [os.path.join(data_args.data_cache_dir, file_name) for file_name in
                     [f'data{index}', f'raw_datasets{index}', f'raw_datasets{index}/tokenized.arrow', f'raw_datasets{index}/grouped.arrow']]

    def tokenize_function(examples):
        output = tokenizer(['<s>' + item + '</s>' for item in examples[text_column_name]])
        return output

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size + block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        logger.info("组合文本输入示例长度%d，分组后大小%d" % (len(examples['input_ids']), len(result["input_ids"])))
        result["labels"] = result["input_ids"].copy()
        return result

    if my_files is not None:
        extension = my_files[0].split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args = {"keep_linebreaks": data_args.keep_linebreaks}
        else:
            dataset_args = {}
    try:
        lm_datasets = datasets.load_from_disk(file_or_names[0], keep_in_memory=False)
    except Exception:
        raw_datasets = load_dataset(
            extension,
            data_files=my_files,
            keep_in_memory=False,
            cache_dir=file_or_names[1],
            **dataset_args,
        )
        # 数据处理和映射
        column_names = list(raw_datasets["train"].features)
        text_column_name = "text" if "text" in column_names else column_names[0]

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            keep_in_memory=False,
            num_proc=data_args.preprocessing_num_workers,
            cache_file_names={k: file_or_names[2] for k in raw_datasets},
            remove_columns=column_names,
            load_from_cache_file=True,
        )

        lm_datasets = tokenized_datasets.map(group_texts, batched=True,
                                             num_proc=data_args.preprocessing_num_workers,
                                             load_from_cache_file=True,
                                             cache_file_names={
                                                 k: file_or_names[3] for k in
                                                 tokenized_datasets},
                                             keep_in_memory=False,
                                             batch_size=80000)

        lm_datasets.save_to_disk(file_or_names[0])
        try:
            shutil.rmtree(file_or_names[1])
        except Exception:
            print("删除目录失败")
        del tokenized_datasets
        del raw_datasets

    return lm_datasets


def _mp_fn(index):
    device = xm.xla_device()

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
    logger.warning(
        f"进程rank: {training_args.local_rank}，设备: {training_args.device}"
        + f"分布式训练: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # 加载分词器
    logger.info('开始加载分词器')
    tokenizer_kwargs = {
        "use_fast": model_args.use_fast_tokenizer,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    tokenizer.add_eos_token = True
    logger.info('加载分词器结束')
    set_seed(training_args.seed)

    lm_datasets = preprocess_data(index,data_args, tokenizer)

    train_dataset = lm_datasets['train']
    logger.info(f'进程{index}开始加载模型')
    config = AutoConfig.from_pretrained(model_args.pretrained_model_name, pretraining_tp=1)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.pretrained_model_name,
        config=config,
        from_tf=bool(".ckpt" in model_args.pretrained_model_name),
    ).to(device)

    for name, param in model.named_parameters():
        if "model.embed_tokens" not in name:
            param.requires_grad = False

    # optimizer =torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()))
    logger.info(f'进程{index}加载模型结束')

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    trainer = init_trainer(model, training_args, train_dataset, tokenizer, log_level)
    # 添加早停回调
    early_stopping_callback = EarlyStoppingCallback(trainer, early_stopping_patience=3)
    trainer.add_callback(early_stopping_callback)

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    logger.info('开始训练')
    # for inputs in train_dataloader:
    #     optimizer.zero_grad()
    #     outputs = model(inputs)
    #     loss = criterion(outputs, targets)
    #     loss.backward()
    #     xm.optimizer_step(optimizer)
    #     xm.mark_step()
    # logger.info('训练结束')
    # torch.save(model.state_dict(), 'model.pth')
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


# def get_strategy():
#     resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
#     tf.config.experimental_connect_to_cluster(resolver)
#     tf.tpu.experimental.initialize_tpu_system(resolver)
#     return tf.distribute.TPUStrategy(resolver)


if __name__ == "__main__":
    # strategy = get_strategy()

    # with strategy.scope():
    #     main()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    xmp.spawn(_mp_fn, args=(), start_method='fork')
