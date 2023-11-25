import logging
import math
import os
import tempfile
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Union, Sequence, Dict

import torch
from datasets import load_dataset, concatenate_datasets,load_from_disk
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict,PeftModel
from peft.tuners.lora import LoraLayer
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    TrainingArguments,
    HfArgumentParser,
    Trainer,
    BitsAndBytesConfig,
    TrainerCallback,
    EarlyStoppingCallback,
    PreTrainedTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)
IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    pretrained_model_name: Optional[str] = field(default="/root/LiteraryAlpaca2")
    tokenizer_name: Optional[str] = field(default="/root/LiteraryAlpaca2")
    use_fast_tokenizer: bool = field(default=True)
    # cache_dir: Optional[str] = field(default='/root/autodl-tmp')


@dataclass
class DataTrainingArguments:
    train_files: Optional[List[str]] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    block_size: Optional[int] = field(default=None)
    preprocessing_num_workers: Optional[int] = field(default=None)
    data_cache_dir: Optional[str] = field(default=None)


@dataclass
class MyTrainingArguments(TrainingArguments):
    double_quant: Optional[bool] = field(default=True)
    quant_type: Optional[str] = field(default="nf4")
    load_in_kbits: Optional[int] = field(default=16)
    max_steps: Optional[int] = field(default=40)
    output_dir: Optional[str] = field(default=tempfile.mkdtemp())
    peft_path: Optional[str] = field(default=None)
    evaluation_strategy: Optional[str] = field(default="steps")
    save_strategy: Optional[str] = field(default="steps")


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: PreTrainedTokenizer
    # 用于处理输入实例并返回模型训练所需的张量字典
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 从每个实例中提取 input_ids 和 labels，并组成元组
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        #对 input_ids 进行批次内的填充
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        # 对 labels 也进行批次内的填充，并用 -100 进行填充
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        # 返回包含 input_ids、labels 和 attention_mask（不等于 pad_token_id 的位置为 1）的字典
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            if 'loss' in state.log_history[-1]:
                print(f"Step: {state.global_step}, Loss: {state.log_history[-1]['loss']}")
            else:
                print(f"Warning: log_history中没有 train_loss 键， 可用 keys: {state.log_history[-1].keys()}")


def build_instruction_dataset(data_path: Union[List[str], str],
                              tokenizer: PreTrainedTokenizer,
                              max_seq_length: int, data_cache_dir=None,
                              preprocessing_num_workers=None,
                              ):
    def tokenization(examples):
        # 定义提示语句的模板
        prompt = (
            "[INST] <<SYS>>\n"
            "You are a helpful assistant. 你是一个乐于助人的助手。你拥有非常丰富的知识。\n"
            "<</SYS>>\n\n{instruction} [/INST]"
        )
        sources = []
        targets = []

        # 遍历输入的数据示例，组合成源文本和目标文本
        for instruction, input_text, output in zip(examples['instruction'], examples['input'], examples['output']):
            if input_text is not None and input_text != "":
                instruction = instruction + '\n' + input_text
            source = prompt.format_map({'instruction': instruction})
            target = f"{output}{tokenizer.eos_token}"

            sources.append(source)
            targets.append(target)

        # 使用分词器对源文本和目标文本进行分词
        tokenized_sources = tokenizer(sources, return_attention_mask=False)
        tokenized_targets = tokenizer(targets, return_attention_mask=False, add_special_tokens=False)

        all_input_ids = []
        all_labels = []

        # 遍历分词后的源文本和目标文本，将其转换为模型输入所需的张量形式
        for s, t in zip(tokenized_sources['input_ids'], tokenized_targets['input_ids']):
            input_ids = torch.LongTensor(s + t)[:max_seq_length]
            labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
            assert len(input_ids) == len(labels)
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        results = {'input_ids': all_input_ids, 'labels': all_labels}
        return results

    all_datasets = []

    # 如果传入的数据路径不是列表，将其转化为列表
    if not isinstance(data_path, (list, tuple)):
        data_path = [data_path]

    try:
        all_datasets = load_from_disk(data_cache_dir)
        logger.info(f'数据集从磁盘加载')
    except Exception:
        for file in data_path:
            # 如果未指定数据缓存目录，将其设为文件所在目录
            if data_cache_dir is None:
                data_cache_dir = str(os.path.dirname(file))
            # 构建缓存文件路径
            cache_path = os.path.join(data_cache_dir, os.path.basename(file).split('.')[0] + f"_{max_seq_length}")
            # 创建缓存目录，如果不存在的话
            os.makedirs(cache_path, exist_ok=True)

            raw_dataset = load_dataset("json", data_files=file, cache_dir=cache_path)
            tokenization_func = tokenization
            tokenized_dataset = raw_dataset.map(
                tokenization_func,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=["instruction", "input", "output"],
                keep_in_memory=False,
            )
            processed_dataset = tokenized_dataset
            # 将数据集格式设置为 PyTorch 格式
            processed_dataset.set_format('torch')
            all_datasets.append(processed_dataset['train'])

        all_datasets = concatenate_datasets(all_datasets)
        all_datasets= all_datasets.train_test_split(test_size=0.1)
        all_datasets.save_to_disk(data_cache_dir)
    return all_datasets


def check_and_create_dir(dir_path):
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True)


def init_trainer(model, training_args, train_dataset, eval_dataset, tokenizer,data_collator):
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
        eval_dataset=eval_dataset,
        data_collator=data_collator,
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
    print('开始加载分词器')
    tokenizer_kwargs = {
        "use_fast": model_args.use_fast_tokenizer,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    # tokenizer.add_eos_token = True
    tokenizer.pad_token = tokenizer.eos_token
    print('加载分词器结束')

    files = [str(file) for path in data_args.train_files for file in Path(path).glob("*.json")]
    print('文件列表：',files)
    dataset_paths = build_instruction_dataset(
        data_path=files,
        tokenizer=tokenizer,
        max_seq_length=data_args.block_size,
        data_cache_dir=data_args.data_cache_dir,
        preprocessing_num_workers=data_args.preprocessing_num_workers
    )
    # dataset_paths=dataset_paths['test'].train_test_split(test_size=0.1)
    train_dataset =dataset_paths['train']
    # eval_dataset =dataset_paths['test'].train_test_split(test_size=0.1)['test']
    eval_dataset =dataset_paths['test']
    print(f"训练数据长度：{len(train_dataset)}")
    print(f"评估数据长度：{len(eval_dataset)}")
    print(tokenizer.decode(train_dataset[0]['input_ids']))
    # config = AutoConfig.from_pretrained(model_args.pretrained_model_name, pretraining_tp=1)

    print('开始加载模型')
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    load_in_4bit = training_args.load_in_kbits == 4
    load_in_8bit = training_args.load_in_kbits == 8
    quantization_config = None
    if training_args.load_in_kbits in [4, 8]:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            llm_int8_threshold=6.0,
            bnb_4bit_compute_dtype=compute_dtype,
            load_in_8bit_skip_modules=['embed_tokens', 'lm_head'],
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type
        )
    if quantization_config is not None:
        logger.info(f"量化配置:{quantization_config.to_dict()}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.pretrained_model_name,
        # config=config,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        quantization_config=quantization_config,
    )
    print('加载模型结束')
    if training_args.load_in_kbits in [4, 8]:
        loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
        for name, param in model.named_parameters():
            if "model.embed_tokens" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
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
        
    if training_args.peft_path is not None:
        model = PeftModel.from_pretrained(model, training_args.peft_path)
    else:
        peft_config = LoraConfig(
            task_type='CAUSAL_LM',
            inference_mode=False,
            r=64, lora_alpha=128,
            target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj'],
            modules_to_save=['embed_tokens', 'lm_head'],
            lora_dropout=0.05)
        model = get_peft_model(model, peft_config)

    # for name, param in model.named_parameters():
    #     print('\n requires_grad属性2:', name, param.requires_grad, "\n")
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if training_args.bf16:
                module = module.to(torch.bfloat16)
            if training_args.fp16:
                module = module.to(torch.float16)
        if 'norm' in name:
            module = module.to(torch.float16)
        if 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
                if training_args.fp16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.float16)
    model.print_trainable_parameters()
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    trainer = init_trainer(model, training_args, train_dataset, eval_dataset, tokenizer,data_collator)
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
    print('开始训练')
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    if not training_args.push_to_hub:
        trainer.save_model()

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    if eval_dataset is not None:
        logger.info("开始评估")

        # 执行评估
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        # 记录评估指标
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
