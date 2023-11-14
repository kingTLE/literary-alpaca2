<p align="center"> <img src="https://github.com/kingTLE/kingTLE/blob/main/header.png"/> </p>

<h1 align="center">
  Literary-Alpaca2
</h1>
<p align="center">
  <font face="黑体" color=orange size="6"> 从词表到微调这就是你需要的一切 </font>
</p>

</br></br>


## 🗂️ 使用指南
- [🔥 项目介绍](#-项目介绍)
- [📝 训练数据](#-训练数据)
- [⏬ 模型部署](#-模型部署)
  - [模型下载](#模型下载)
    - [基于Llama2的中文预训练模型](#基于Llama2的中文预训练模型)
    - [基于LiteraryAlpaca2的中文微调模型Chat](#基于LiteraryAlpaca2的中文微调模型Chat)
  - [模型调用代码示例](#模型调用代码示例)
  - [Gradio快速搭建问答平台](#gradio快速搭建问答平台)
- [词表训练](#词表训练)
- [预训练](#预训练)
- [微调](#微调)
  - [数据准备](#数据准备)
  - [微调脚本](#微调脚本)
- [参考论文](#参考论文)



## 🔥 项目介绍


本仓库将展示如何从词表开始构建自己的词表与使用基座模型预训练和微调模型
仓库中的代码示例主要是基于Llama2的Hugging Face版本进行训练。


## 📝 训练数据
| 类型                                                       | 描述                                                         |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| 网络小说                                                   | 高质量长文本数据 |
| [Math23K](https://opendatalab.org.cn/Math23K)               | 中文数学问题                                          |
| [LCCC](https://github.com/thu-coai/CDial-GPT)               | 中文开源对话集                                       |

</br></br>
词表与预训练阶段数据对比图：
<p align="center"> <img src="img/data_comparison.png" width=80%/> </p>

## ⏬ 模型部署

<p>Meta官方的下载链接：https://huggingface.co/meta-llama</p>

中文预训练模型、LoRA参数、chat模型都已上传至[Hugging Face](https://huggingface.co/taotie1) 目前只有13B模型。

### 模型下载

#### 基于Llama2的中文预训练模型

|  类别        | 🤗模型名称   | 基座模型          |   下载地址          |
|  ----------  | ---------- |  ----------------- | ------------------- |
|  预训练 | taotie1/literary-alpaca2-13B |     meta-llama/Llama-2-13b-hf     |[模型下载](https://huggingface.co/taotie1/literary-alpaca2-13B) |
|  LoRA | taotie1/literary-alpaca2-13B-lora |      taotie1/literary-alpaca2-13B     |[模型下载](https://huggingface.co/taotie1/literary-alpaca2-13B-lora) |
#### 基于LiteraryAlpaca2的中文微调模型Chat
|  类别           | 🤗模型名称        | 下载地址                                                 |
| --------------- | ---------------    |  ------------------------------------------------------------ |
|  Chat  |  taotie1/literary-alpaca2-13B-chat  | [模型下载](https://huggingface.co/taotie1/literary-alpaca2-13B-chat) |


### 模型调用代码示例
根据[requirements.txt](https://github.com/kingTLE/literary-alpaca2/blob/main/requirements.txt)安装环境依赖
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('taotie1/literary-alpaca2-13B-chat',device_map='auto',torch_dtype=torch.float16,load_in_8bit=True)
model =model.eval()
tokenizer = AutoTokenizer.from_pretrained('taotie1/literary-alpaca2-13B-chat',use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
input_ids = tokenizer(['<s>Human: 介绍一下中国\n</s><s>Assistant: '], return_tensors="pt",add_special_tokens=False).input_ids.to('cuda')        
generate_input = {
    "input_ids":input_ids,
    "max_new_tokens":512,
    "do_sample":True,
    "top_k":50,
    "top_p":0.95,
    "temperature":0.3,
    "repetition_penalty":1.3,
    "eos_token_id":tokenizer.eos_token_id,
    "bos_token_id":tokenizer.bos_token_id,
    "pad_token_id":tokenizer.pad_token_id
}
generate_ids  = model.generate(**generate_input)
text = tokenizer.decode(generate_ids[0])
print(text)
```

### Gradio快速搭建问答平台

基于gradio搭建的问答界面，实现了流式的输出，将下面代码复制到控制台运行，以下代码以Atom-7B模型为例，<font color="#006600">不同模型只需修改一下代码里的模型名称就好了😊</font><br/>
```
python examples/chat_gradio.py --model_name_or_path FlagAlpha/Atom-7B
```
## 词表训练

先对你的训练数据进行[命名清洗](https://github.com/kingTLE/literary-alpaca2/tree/main/chinese-tokenizer/Batch_Rename.py)【可选】</br></br>
选择运行[随机清洗代码](https://github.com/kingTLE/literary-alpaca2/tree/main/chinese-tokenizer/random_sample.py)或[全部清洗](https://github.com/kingTLE/literary-alpaca2/tree/main/chinese-tokenizer/clear.py)，在[ill_ocr_regex.txt](https://github.com/kingTLE/literary-alpaca2/tree/main/chinese-tokenizer/ill_ocr_regex.txt)中可以自定义你的正则。

运行[full_sample_extraction.py](https://github.com/kingTLE/literary-alpaca2/tree/main/chinese-tokenizer/full_sample_extraction.py)把数据合并成一个文件。

参照[train-chinese-tokenizer.ipynb](https://github.com/kingTLE/literary-alpaca2/tree/main/chinese-tokenizer/train-chinese-tokenizer.ipynb)进行词表训练，可以根据自己的需求修改代码。
训练完成后把你的词表放入my-tokenizer目录下。按照下面方式和原llama2的tokenizer合并
```
bash运行
'
Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python incorporation.py
'
```
运行[text.py](https://github.com/kingTLE/literary-alpaca2/tree/main/chinese-tokenizer/text.py)进行测试词表效果

## 预训练
本仓库训练代码使用[DeepSpeed](https://github.com/microsoft/DeepSpeed)加速
- 模型预训练脚本：[train/GPU/pretrain-peft1.sh](https://github.com/kingTLE/literary-alpaca2/tree/main/train/GPU/pretrain-peft1.sh)
- 预训练实现代码：[train/GPU/pretrain-peft1.py](https://github.com/kingTLE/literary-alpaca2/tree/main/train/GPU/pretrain-peft1.py)


## 微调

### 数据准备
在data目录下提供了一份用于模型sft的数据样例：
- 训练数据：[data/train_sft.csv](https://github.com/FlagAlpha/Llama2-Chinese/blob/main/data/train_sft.csv)
- 验证数据：[data/dev_sft.csv](https://github.com/FlagAlpha/Llama2-Chinese/blob/main/data/dev_sft.csv)

每个csv文件中包含一列“text”，每一行为一个训练样例，每个训练样例按照以下格式将问题和答案组织为模型输入，您可以按照以下格式自定义训练和验证数据集：
```
"<s>Human: "+问题+"\n</s><s>Assistant: "+答案
```
例如，
```
<s>Human: 用一句话描述地球为什么是独一无二的。</s><s>Assistant: 因为地球是目前为止唯一已知存在生命的行星。</s>
```
### 微调脚本

LoRA微调脚本见：[train/sft/finetune_lora.sh](https://github.com/FlagAlpha/Llama2-Chinese/blob/main/train/sft/finetune_lora.sh)，关于LoRA微调的具体实现代码见[train/sft/finetune_clm_lora.py](https://github.com/FlagAlpha/Llama2-Chinese/blob/main/train/sft/finetune_clm_lora.py)，单机多卡的微调可以通过修改脚本中的`--include localhost:0`来实现。




## 参考论文

* [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
* [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
* [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
* [Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca](https://arxiv.org/abs/2304.08177)

<p align="center" width="100%">
<img src="https://starchart.cc/kingTLE/literary-alpaca2.svg" alt="Star History" style="width: 100%; display: block; margin: auto;">
</p>
