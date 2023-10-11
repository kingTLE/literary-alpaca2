import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

import datasets
import torch
import transformers
from datasets import load_dataset
from torchdata.datapipes.iter import IterableWrapper
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    set_seed,
    TrainingArguments,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    HfArgumentParser,
    default_data_collator,
    BitsAndBytesConfig,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from peft.tuners.lora import LoraLayer
logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# 添加 ModelArguments 数据类
@dataclass
class ModelArguments:
    # 定义模型参数
    pretrained_model_name: Optional[str] = field(
        default="/kaggle/input/llama-2/pytorch/13b-hf/1",
        metadata={"help": "预训练模型的名称"},
    )
    tokenizer_name: Optional[str] = field(
        default="/kaggle/input/merged-tokenizer/incorporation_hf2",
        metadata={"help": "预训练分词器的名称或路径，如果与model_name不同"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "是否使用快速分词器"},
    )
    torch_dtype: Optional[str] = field(default=None)
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "将使用在运行'huggingface-cli login'时生成的令牌（与私有模型一起使用时必需）."},
    )


# 添加 DataTrainingArguments 数据类
@dataclass
class DataTrainingArguments:
    # 定义数据训练参数
    train_files: Optional[List[str]] = field(
        default=None, metadata={"help": "训练数据的列表"},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "在使用TXT文件时是否保留换行符"}
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={"help": "训练数据集将以此大小的块进行截断以进行训练"},
    )


@dataclass
class MyTrainingArguments(TrainingArguments):
    modules_to_save: Optional[str] = field(default=None)
    double_quant: Optional[bool] = field(default=True)
    quant_type: Optional[str] = field(default="nf4")
    load_in_kbits: Optional[int] = field(default=16)
    max_steps: Optional[int] = field(default=None)


def main():
    # 加载模型和数据
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    send_example_telemetry("run_clm", model_args, data_args)

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

    set_seed(training_args.seed)

    if True:
        data_files = {}
        dataset_args = {}
        if data_args.train_files is not None:
            data_files["train"] = data_args.train_files
            print('训练文件总个数', len(data_args.train_files))
        if data_files["train"] is not None:
            extension = data_files["train"][0].split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            streaming=True,
            cache_dir=os.path.join(training_args.output_dir, 'dataset_cache'),
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )
        raw_datasets = raw_datasets.shuffle(seed=training_args.seed, buffer_size=1000000)
    dataset_head = raw_datasets["train"].take(3)
    print(list(dataset_head))
    column_names = list(list(dataset_head)[0].keys())

    print(column_names)
    text_column_name = "text" if "text" in column_names else column_names[0]

    print(training_args.local_rank, '开始加载分词器')
    tokenizer_kwargs = {
        "use_fast": model_args.use_fast_tokenizer,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    tokenizer.add_eos_token = True
    print(training_args.local_rank, '加载分词器结束')

    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    config_kwargs = {
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    config = AutoConfig.from_pretrained(model_args.pretrained_model_name, **config_kwargs)

    print(training_args.local_rank, '开始加载模型')
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    if training_args.load_in_kbits in [4, 8]:
        load_in_4bit = training_args.load_in_kbits == 4
        load_in_8bit = training_args.load_in_kbits == 8

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=training_args.load_in_kbits == 4,
            load_in_8bit=training_args.load_in_kbits == 8,
            llm_int8_threshold=6.0,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type
        )
    else:
        load_in_4bit = False
        load_in_8bit = False
        quantization_config = None
    if quantization_config is not None:
        logger.info(f"量化配置:{quantization_config.to_dict()}")
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = AutoModel.from_pretrained(
        model_args.pretrained_model_name,
        config=config,
        from_tf=bool(".ckpt" in model_args.pretrained_model_name),
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch_dtype,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    )
    print(training_args.local_rank, '加载模型结束')
    if training_args.load_in_kbits in [4, 8]:
        loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
        for param in model.parameters():
            if (param.dtype == torch.float16)  and loaded_in_kbit:
                param.data = param.data.to(torch.float32)
        for name, param in model.named_parameters():
            if "model.embed_tokens" not in name:
                param.requires_grad = False
            if 'norm' in name:
                module = param.to(torch.float32)
        if loaded_in_kbit and training_args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, _input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            model.gradient_checkpointing_enable()

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if training_args.fp16:
                module = module.to(torch.float16)
        if 'norm' in name:
            module = module.to(torch.float16)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if training_args.fp16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.float16)
    if training_args.fp16:
        for name, module in model.named_modules():
            if name.startswith('model'):
                module.to(torch.float16)


    model.config.use_cache = False
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=16)

    def tokenize_function(examples):
        with CaptureLogger(tok_logger):
            output = tokenizer(['<s>' + item + '</s>' for item in examples[text_column_name]])
        return output

    with training_args.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            batch_size=32
        )
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "所选的标记器支持的`model_max_length`长度超过了默认的`block_size`值1024。"
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"传入的block_size({data_args.block_size})大于模型的最大长度({tokenizer.model_max_length})。"
                f"使用block_size={tokenizer.model_max_length}。"
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with training_args.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            batch_size=32,
        )
    train_dataset = lm_datasets['train']

    print(training_args.local_rank, '初始化Trainer')
    train_dataset_length = 0
    for _ in train_dataset:
        train_dataset_length += 1

    max_steps = (
        training_args.max_steps if training_args.max_steps is not None else train_dataset_length * training_args.num_train_epochs
    )
    training_args.max_steps = max_steps
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=IterableWrapper(train_dataset),
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    print(training_args.local_rank, '结束初始化Trainer')

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    print(training_args.local_rank, '开始训练')
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else train_dataset_length
    )
    metrics["train_samples"] = min(max_train_samples, train_dataset_length)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
