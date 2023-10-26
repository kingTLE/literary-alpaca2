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
import torch
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
    BitsAndBytesConfig,
    TrainerCallback
)
from transformers.trainer_utils import get_last_checkpoint

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
    double_quant: Optional[bool] = field(default=True)
    quant_type: Optional[str] = field(default="nf4")
    load_in_kbits: Optional[int] = field(default=16)
    max_steps: Optional[int] = field(default=2)
    load_best_model_at_end: Optional[bool] = field(default=True)
    output_dir: Optional[str] = field(default=tempfile.mkdtemp())
    run_name: Optional[str] = field(default=output_dir)
    evaluation_strategy: Optional[str] = field(default="steps")
    save_strategy: Optional[str] = field(default="steps")


class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            print(f"Step: {state.global_step}, Loss: {state.log_history[-1]['loss']}")


def preprocess_data(data_args, tokenizer):
    block_size = min(data_args.block_size, tokenizer.model_max_length) if data_args.block_size else min(1024,
                                                                                                        tokenizer.model_max_length)
    data_files = data_args.train_files
    file_or_names = [os.path.join(data_args.data_cache_dir, file_name) for file_name in
                     ['data', 'raw_datasets', 'raw_datasets/tokenized.arrow', 'raw_datasets/grouped.arrow']]

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
                                             batch_size=32)

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


def init_trainer(model, training_args, train_dataset, tokenizer, log_level):
    max_steps = (
        training_args.max_steps if training_args.max_steps is not None else int((len(
            train_dataset) * training_args.num_train_epochs) / (
                                                                                        16 * training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps))

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
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"进程rank: {training_args.local_rank}，设备: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"分布式训练: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"训练参数 {training_args}")
    set_seed(training_args.seed)
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
    logger.info('加载分词器结束')

    train_dataset = preprocess_data(data_args, tokenizer)['train']
    logger.info(f"数据集长度：{len(train_dataset)}")

    logger.info(tokenizer.decode(train_dataset[0]['input_ids']))
    config = AutoConfig.from_pretrained(model_args.pretrained_model_name, pretraining_tp=1)

    print(training_args.local_rank, '开始加载模型')
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    if training_args.load_in_kbits in [4, 8]:
        load_in_4bit = training_args.load_in_kbits == 4
        load_in_8bit = training_args.load_in_kbits == 8

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=training_args.load_in_kbits == 4,
            load_in_8bit=training_args.load_in_kbits == 8,
            llm_int8_threshold=8.0,
            bnb_4bit_compute_dtype=compute_dtype,
            load_in_8bit_skip_modules='embed_tokens',
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type
        )
    if quantization_config is not None:
        logger.info(f"量化配置:{quantization_config.to_dict()}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.pretrained_model_name,
        config=config,
        from_tf=bool(".ckpt" in model_args.pretrained_model_name),
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    )
    print(training_args.local_rank, '加载模型结束')
    if training_args.load_in_kbits in [4, 8]:
        loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
        for name, param in model.named_parameters():
            if "model.embed_tokens" not in name:
                param.requires_grad = False
        for param in model.parameters():
            if ((param.dtype == torch.float16) or (param.dtype == torch.bfloat16)) and loaded_in_kbit:
                param.data = param.data.to(torch.float32)
        for name, module in model.named_modules():
            if 'norm' in name:
                module = module.to(torch.float32)
        # if loaded_in_kbit and training_args.gradient_checkpointing:
        #     if hasattr(model, "enable_input_require_grads"):
        #         model.enable_input_require_grads()
        #     else:
        #         def make_inputs_require_grad(module, _input, output):
        #             output.requires_grad_(True)
        #
        #         model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        #     model.gradient_checkpointing_enable()
    model.config.use_cache = False
    embedding_size = model.get_input_embeddings().weight.shape[0]

    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        # , pad_to_multiple_of=16)

    trainer = init_trainer(model, training_args, train_dataset, tokenizer, log_level)
    # 添加早停回调
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)
    trainer.add_callback(early_stopping_callback)

    loss_logging_callback = LossLoggingCallback()
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


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    main()
