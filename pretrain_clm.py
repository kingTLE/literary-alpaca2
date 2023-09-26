import logging
import math
import os
import sys
from dataclasses import dataclass, field
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
from itertools import chain
import deepspeed
from typing import Optional, List

import datasets
import pandas as pd
import evaluate
import torch
from datasets import load_dataset
from datasets.combine import interleave_datasets
import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
import datetime
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from datasets import interleave_datasets

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    用于指定进行微调或从头训练的模型/配置文件/分词器的参数。
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "用于初始化权重的模型检查点。如果要从头开始训练模型，则不设置该参数。"
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "如果从头开始训练模型，请从列表中选择一个模型类型：" + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "在从头训练模型时，用于覆盖一些默认配置设置的字符串。示例："
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "预训练配置文件的名称或路径，如果与model_name不同。"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "预训练分词器的名称或路径，如果与model_name不同。"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "从huggingface.co下载的预训练模型存储的目录。"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "是否使用快速分词器（由tokenizers库支持）."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "要使用的特定模型版本（可以是分支名称、标签名称或提交ID）。"},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "将使用在运行'huggingface-cli login'时生成的令牌（与私有模型一起使用时必需）."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "覆盖默认的'torch.dtype'并使用指定的数据类型加载模型。"
                "如果传递'auto'，数据类型将根据模型的权重自动推导。"
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "不能同时使用'--config_overrides'和'--config_name'或'--model_name_or_path'参数。"
            )


@dataclass
class DataTrainingArguments:
    """
    用于训练和评估模型的数据相关参数。
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "要使用的数据集的名称（通过datasets库）"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "要使用的数据集的配置名称（通过datasets库）"}
    )
    train_files: Optional[List[str]] = field(default=None, metadata={"help": "输入的训练数据文件（文本文件）"})
    validation_files: Optional[List[str]] = field(
        default=None,
        metadata={"help": "用于评估困惑度的可选输入评估数据文件（文本文件）"},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "为了调试或更快地训练，如果设置了该值，则将训练示例的数量截断为此值"
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "为了调试或更快地训练，如果设置了该值，则将评估示例的数量截断为此值"
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "令牌化后的可选输入序列长度"
                "训练数据集将以此大小的块进行截断以进行训练"
                "对于单个句子输入，默认为模型的最大输入长度（考虑特殊标记）"
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "覆盖缓存的训练和评估集"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "如果没有验证集拆分，则训练集中用作验证集的百分比"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "用于预处理的进程数"},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "在使用TXT文件时是否保留换行符"}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "流式特性需要`datasets>=2.0.0`版本")

        if self.dataset_name is None and self.train_files is None and self.validation_files is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_files is not None:
                extension = self.train_files[0].split(".")[-1]
                assert extension in ["csv", "json", "txt"], "train_file`应为csv、json或txt文件"
            if self.validation_files is not None:
                extension = self.validation_files[0].split(".")[-1]
                assert extension in ["csv", "json", "txt"], "validation_file`应为csv、json或txt文件"


def main():
    # 查看所有可能的参数在src/transformers/training_args.py中
    # 或通过在此脚本中传递--help标志来查看。
    # 现在我们保持不同的参数集，以便更清晰地分离责任。

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 如果我们只向脚本传递一个参数，并且它是一个json文件的路径，
        # 让我们解析它以获取我们的参数。
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 发送遥测。跟踪示例用法可以帮助我们更好地分配资源来维护它们。发送的信息是与Python/PyTorch版本一起传递的参数。
    send_example_telemetry("run_clm", model_args, data_args)

    # 设置日志记录
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # training_args.log_level的默认值是passive，所以我们在这里将日志级别设置为info，以便具有该默认值。
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 在每个进程上记录小的摘要：
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 检测最后一个检查点。
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 在初始化模型之前设置种子。
    set_seed(training_args.seed)

    # 获取数据集：您可以提供自己的CSV/JSON/TXT训练和评估文件（参见下文），
    # 或者只需提供Hub上可用的公共数据集之一的名称https://huggingface.co/datasets/
    # （数据集将从数据集Hub自动下载）。
    #
    # 对于CSV/JSON文件，此脚本将使用名为'text'的列或者如果找不到名为'text'的列，则使用第一列。您可以轻松调整此行为（参见下文）。
    #
    # 在分布式训练中，load_dataset函数保证只有一个本地进程可以同时下载数据集。
    if True:
        data_files = {}
        dataset_args = {}
        if data_args.train_files is not None:
            print(data_args.train_files)
            data_files["train"] = data_args.train_files
            print('训练文件总个数', len(data_args.train_files))
        if data_args.validation_files is not None:
            data_files["validation"] = data_args.validation_files
        extension = (
            data_files["train"][0].split(".")[-1]
            if data_files["train"] is not None
            else data_args.validation_files.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks

        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            streaming=data_args.streaming,
            cache_dir=os.path.join(training_args.output_dir, 'dataset_cache'),
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )
        if data_args.streaming:
            raw_datasets = raw_datasets.shuffle(seed=training_args.seed, buffer_size=1000000)
        # 如果没有验证数据，将使用validation_split_percentage来划分数据集。
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )

    # 有关加载任何类型的标准或自定义数据集（从文件、python字典、pandas DataFrame等）的更多信息，请访问
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # 加载预训练模型和分词器
    #
    # 分布式训练：
    # .from_pretrained方法保证只有一个本地进程可以同时下载模型和词汇表。

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    print(training_args.local_rank, '开始加载分词器')
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    print(training_args.local_rank, '加载分词器结束')
    print(training_args.local_rank, '开始加载模型')
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params / 2 ** 20:.2f}M params")
    print(training_args.local_rank, '加载模型结束')
    # 仅在需要时调整嵌入大小，以避免索引错误。如果您从头开始创建一个模型，并且词汇表很小，想要较小的嵌入大小，请删除此测试。
    for name, param in model.named_parameters():
        if "model.embed_tokens" not in name:
            param.requires_grad = False

    embedding_size = model.get_input_embeddings().weight.shape[0]

    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    # 预处理数据集。
    # 首先我们对所有文本进行分词。
    if training_args.do_train:
        if data_args.streaming:
            dataset_head = raw_datasets["train"].take(3)
            print(list(dataset_head))
            column_names = list(list(dataset_head)[0].keys())
        else:
            column_names = list(raw_datasets["train"].features)
    else:
        if data_args.streaming:
            dataset_head = raw_datasets["validation"].take(3)
            column_names = list(list(dataset_head)[0].keys())
        else:
            column_names = list(raw_datasets["validation"].features)
    print(column_names)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # 由于这将被pickled以避免在Hasher中出现_LazyModule错误，因此强制在tokenize_function之前加载logger
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(['<s>' + item + '</s>' for item in examples[text_column_name]])
        return output

    with training_args.main_process_first(desc="数据集映射和标记化"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="在数据集上运行标记器",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
                batch_size=60000,
            )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "所选的标记器支持的`model_max_length`长度超过了默认的`block_size`值1024。如果您想使用更长的`block_size`，"
                "最大可使用`tokenizer.model_max_length`，您可以使用`--block_size xxx`覆盖此默认值。"
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"传入的block_size({data_args.block_size})大于模型的最大长度({tokenizer.model_max_length})。"
                f"使用block_size={tokenizer.model_max_length}。"
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        # concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        logger.info("组合文本输入示例长度%d，分组后大小%d" % (len(examples['input_ids']), len(result["input_ids"])))
        result["labels"] = result["input_ids"].copy()
        return result

    # 请注意，在 `batched=True` 的情况下，此映射会同时处理 1,000 个文本，
    # 因此 group_texts 会丢弃剩余的的余数。您可以在此调整batch_size，但数值越大，预处理速度越慢。
    with training_args.main_process_first(desc="将文本分组在一起"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"将文本分组为大小为{block_size}的块",
                batch_size=40000,
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                batch_size=60000,
            )

    print(training_args.local_rank, '开始选择训练数据集')
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train需要一个训练数据集")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None and data_args.streaming == False:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
    print(training_args.local_rank, '结束选择训练数据集')

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval需要一个验证数据集")
        print(training_args.local_rank, '开始选择评估数据集')
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None and data_args.streaming == False:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        print(training_args.local_rank, '结束选择评估数据集')

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # 根据模型和配置，logits 可能包含额外的张量、 如 past_key_values，但 logits 始终排在前面
                logits = logits[0]
            return logits.argmax(dim=-1)

        print(training_args.local_rank, '开始加载指标')
        metric = evaluate.load("accuracy.py")
        print(training_args.local_rank, '结束加载指标')

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    print(training_args.local_rank, '初始化Trainer')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=IterableWrapper(train_dataset) if training_args.do_train else None,
        eval_dataset=IterableWrapper(eval_dataset) if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        # callbacks=([SavePeftModelCallback] if isinstance(model, PeftModel) else None),
    )
    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        print(training_args.local_rank, '开始训练')
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

    if training_args.do_eval:
        logger.info("*** 评估 ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
