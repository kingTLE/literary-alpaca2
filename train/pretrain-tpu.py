import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Optional, List

import datasets
import torch
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from datasets import load_dataset
from huggingface_hub import HfApi
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser, DataCollatorWithPadding,

)
from transformers.trainer_utils import get_last_checkpoint


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
class MyTrainingArguments:
    output_dir: Optional[str] = field(default=tempfile.mkdtemp())
    hub_token: Optional[str] = field(default=None)
    hub_model_id: Optional[str] = field(default=None)
    report_to: Optional[str] = field(default="tensorboard")
    learning_rate: Optional[float] = field(default=1e-5)
    resume_from_checkpoint: Optional[str] = field(default=None)


def check_and_create_dir(dir_path):
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True)


def preprocess_data(data_args, tokenizer):
    block_size = min(data_args.block_size, tokenizer.model_max_length) if data_args.block_size else min(1024,
                                                                                                        tokenizer.model_max_length)
    data_files = data_args.train_files

    file_or_names = [os.path.join(data_args.data_cache_dir, file_name) for file_name in
                     ['data', 'raw_datasets', 'raw_datasets/tokenized.arrow', 'raw_datasets/grouped.arrow']]

    def tokenize_function(examples):
        output = tokenizer(['<s>' + item + '</s>' for item in examples[text_column_name]],truncation=True, max_length=block_size)
        return output

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        logger.info("组合文本输入示例长度%d，分组后大小%d" % (len(examples['input_ids']), len(result["input_ids"])))
        result["labels"] = result["input_ids"].copy()
        return result

    if data_files is not None:
        extension = data_files[0].split(".")[-1]
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
            data_files=data_files,
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
                                             batch_size=8)

        lm_datasets.save_to_disk(file_or_names[0])
        try:
            shutil.rmtree(file_or_names[1])
        except Exception:
            print("删除目录失败")
        try:
            del tokenized_datasets
            del raw_datasets
        except Exception:
            print("变量已被删除")

    return lm_datasets


def train(index, FLAGS):
    device = xm.xla_device()
    # 加载模型和数据
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=8, rank=xm.get_ordinal(),
                                                                    shuffle=True)
    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS['BATCH_SIZE'], collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding=True),
                                                  sampler=train_sampler)
    xla_train_loader = pl.MpDeviceLoader(training_loader, device)
    logger.info(f'进程{index}开始加载模型')
    # config = AutoConfig.from_pretrained(model_args.pretrained_model_name, pretraining_tp=1)
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint if checkpoint is not None else model_args.pretrained_model_name,
        # config=config,
        torch_dtype=torch.bfloat16,
    ).to(device)
    for name, param in model.named_parameters():
        param.requires_grad = False
        if "model.embed_tokens" in name:
            param.requires_grad = True
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    logger.info(f'进程{index}加载模型结束')

    logger.info('开始训练')
    lr = training_args.learning_rate
    model.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                                                  total_iters=FLAGS['NUM_STEPS'] * FLAGS['BATCH_SIZE'])
    num_iterations = int(FLAGS['NUM_STEPS'] / FLAGS['BATCH_SIZE'] / 8)
    for epoch in range(1, FLAGS['NUM_EPOCHS'] + 1):
        for step, data in enumerate(xla_train_loader):
            optimizer.zero_grad()
            outputs = model(**data)
            loss = outputs.loss
            loss.backward()
            xm.optimizer_step(optimizer)
            if (step + 1) % FLAGS['LOGGING_STEPS'] == 0:
                xm.master_print(
                    f'Loss: {loss.item()}, {step + 1} steps out of {num_iterations}, LR: {optimizer.param_groups[0]["lr"]}')
            scheduler.step()
        xm.master_print(f"Trained for {epoch} epochs out of {FLAGS['NUM_EPOCHS']}")
        xm.master_print("Waiting for all processes across cores to finish")
        xm.rendezvous('init')
        xm.master_print("Saving the model")
        xm.save(model.state_dict(training_args.output_dir), "pytorch_model.bin")


if __name__ == "__main__":
    os.environ.pop('TPU_PROCESS_ADDRESSES')
    os.environ.pop('CLOUD_TPU_TASK_ID')
    os.environ['XLA_USE_BF16'] = "1"
    MAX_INPUT = 128
    logger = logging.getLogger(__name__)
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    check_and_create_dir(training_args.output_dir)
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
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
    train_dataset = preprocess_data(data_args, tokenizer)['train']

    print('长度：', len(train_dataset))
    FLAGS = {'MAX_INPUT': 128,
             'LOGGING_STEPS': 100,
             'NUM_EPOCHS': 1,
             'BATCH_SIZE': 8,
             'NUM_STEPS': len(train_dataset)}
    xmp.spawn(train, args=(FLAGS,), start_method='fork')
    login(training_args.hub_token)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=training_args.output_dir,
        path_in_repo="pytorch_model.bin",
        repo_id=training_args.hub_model_id,
        repo_type="model",
        create_pr=1
    )
