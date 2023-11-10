import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List
from pathlib import Path
import logging
import datasets
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    TrainingArguments,
    HfArgumentParser,
    Trainer,
    default_data_collator,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    pretrained_model_name: Optional[str] = field(default="/root/autodl-tmp/Llama-2-13b-hf")
    tokenizer_name: Optional[str] = field(default="/root/autodl-fs/train/incorporation_hf2")
    use_fast_tokenizer: bool = field(default=True)
    # cache_dir: Optional[str] = field(default='/root/autodl-tmp')

@dataclass
class DataTrainingArguments:
    train_files: Optional[List[str]] = field(default=None)
    keep_linebreaks: bool = field(default=True)
    block_size: Optional[int] = field(default=None)
    preprocessing_num_workers: Optional[int] = field(default=None)
    data_cache_dir: Optional[str] = field(default="/root/autodl-fs/train/working/dataset_cache")
    max_train_samples: Optional[int] = field(default=None)

@dataclass
class MyTrainingArguments(TrainingArguments):
    double_quant: Optional[bool] = field(default=True)
    quant_type: Optional[str] = field(default="nf4")
    load_in_kbits: Optional[int] = field(default=16)
    max_steps: Optional[int] = field(default=47)
    load_best_model_at_end: Optional[bool] = field(default=False)
    output_dir: Optional[str] = field(default=tempfile.mkdtemp())
    run_name: Optional[str] = field(default=output_dir)
    evaluation_strategy: Optional[str] = field(default="no")
    save_strategy: Optional[str] = field(default="steps")


class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            if 'loss' in state.log_history[-1]:
                print(f"Step: {state.global_step}, Loss: {state.log_history[-1]['loss']}")
            else:
                print(f"Warning: log_history中没有 train_loss 键， 可用 keys: {state.log_history[-1].keys()}")

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
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
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
                                             batch_size=80000)

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


def check_and_create_dir(dir_path):
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True)


def init_trainer(model, training_args, train_dataset, tokenizer):
    max_steps = (
        training_args.max_steps if training_args.max_steps is not None else int((len(
            train_dataset) * training_args.num_train_epochs) / (
                                                                                    training_args.per_device_train_batch_size))

    )
    training_args.max_steps = max_steps
    print('开始初始化Trainer')
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
    print(f'结束初始化Trainer，用时 {elapsed_time:.2f} 秒')
    return trainer


def main():
    # 加载模型和数据
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger.warning(
        f"进程rank: {training_args.local_rank}，设备: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"分布式训练: {bool(training_args.local_rank != -1)}"
    )
    logger.info(f"训练参数 {training_args}")
    set_seed(training_args.seed)
    # check_and_create_dir(training_args.output_dir)
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
    print('开始加载分词器')
    tokenizer_kwargs = {
        "use_fast": model_args.use_fast_tokenizer,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    tokenizer.add_eos_token = True
    tokenizer.pad_token = tokenizer.eos_token
    print('加载分词器结束')

    train_dataset = preprocess_data(data_args, tokenizer)['train']
    logger.info(f"数据集长度：{len(train_dataset)}")

    logger.info(tokenizer.decode(train_dataset[0]['input_ids']))
    # config = AutoConfig.from_pretrained(model_args.pretrained_model_name, pretraining_tp=1)

    print('开始加载模型')
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    load_in_4bit = training_args.load_in_kbits == 4
    load_in_8bit = training_args.load_in_kbits == 8
    if training_args.load_in_kbits in [4, 8]:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=training_args.load_in_kbits == 4,
            load_in_8bit=training_args.load_in_kbits == 8,
            llm_int8_threshold=8.0,
            bnb_4bit_compute_dtype=compute_dtype,
            load_in_8bit_skip_modules='embed_tokens',
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type
        )
    # if quantization_config is not None:
    #     logger.info(f"量化配置:{quantization_config.to_dict()}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.pretrained_model_name,
        # config=config,
        # cache_dir=model_args.cache_dir,
        from_tf=bool(".ckpt" in model_args.pretrained_model_name),
        # token=training_args.hub_token,
        # torch_dtype=torch.bfloat16,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        # quantization_config=quantization_config,
    )
    print('加载模型结束')
    if training_args.load_in_kbits in [4, 8]:
        loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
        for name, param in model.named_parameters():
            param.requires_grad = False
        for param in model.parameters():
            if ((param.dtype == torch.float16) or (param.dtype == torch.bfloat16)) and loaded_in_kbit:
                param.data = param.data.to(torch.float32)
        for name, module in model.named_modules():
            if 'norm' in name:
                module = module.to(torch.float32)
        if loaded_in_kbit and training_args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, _input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            model.gradient_checkpointing_enable()
    model.config.use_cache = False
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    # model = prepare_model_for_kbit_training(model)
    # peft_config = LoraConfig(
    #     task_type='CAUSAL_LM',
    #     inference_mode=False,
    #     r=64, lora_alpha=128,
    #     target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj'],
    #     modules_to_save=["embed_tokens"],
    #     lora_dropout=0.05)
    # model = get_peft_model(model, peft_config)
    for name, param in model.named_parameters():
        if "model.embed_tokens" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
        print('\n requires_grad属性2:', name, param.requires_grad,"\n")
    # for name, module in model.named_modules():
    #     if isinstance(module, LoraLayer):
    #         if training_args.bf16:
    #             module = module.to(torch.bfloat16)
    #         if training_args.fp16:
    #             module = module.to(torch.float16)
    #     if 'norm' in name:
    #         module = module.to(torch.float16)
    #     if 'embed_tokens' in name:
    #         if hasattr(module, 'weight'):
    #             if training_args.bf16 and module.weight.dtype == torch.float32:
    #                 module = module.to(torch.bfloat16)
    #             if training_args.fp16 and module.weight.dtype == torch.float32:
    #                 module = module.to(torch.float16)
    # model.print_trainable_parameters()
    trainer = init_trainer(model, training_args, train_dataset, tokenizer)
    # 添加早停回调
    # early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)
    # trainer.add_callback(early_stopping_callback)

    loss_logging_callback = LossLoggingCallback()
    trainer.add_callback(loss_logging_callback)
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    print('开始训练')
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    if not training_args.push_to_hub:
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
    main()
