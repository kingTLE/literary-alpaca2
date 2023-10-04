"""
bash
'
Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python incorporation.py
'
"""

import os
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as pb2_model
import sentencepiece as spm

llama_tokenizer_dir = "./incorporation_hf"
chinese_model_file = "my-tokenizer/chinese-tokenizer.model"
output_sp_dir = 'incorporation_sp'
output_hf_dir = 'incorporation_hf'


def load_tokenizers(llama_tokenizer_dir, chinese_model_file):
    llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
    chinese_model = spm.SentencePieceProcessor()
    chinese_model.Load(chinese_model_file)
    return llama_tokenizer, chinese_model


def print_token_info(tokenizer):
    if isinstance(tokenizer, LlamaTokenizer):
        print("llama_tokenizer-Token 数量:", len(tokenizer))
        print("特殊 token:", tokenizer.all_special_tokens)
        print("特殊 token 的 ID:", tokenizer.all_special_ids)
        print("特殊 token 映射:", tokenizer.special_tokens_map)
    elif isinstance(tokenizer, spm.SentencePieceProcessor):
        print("my-tokenizer的Token 数量:", tokenizer.get_piece_size())
    else:
        print("无法识别的分词器类型")


def save_merged_tokenizer(llama_tokenizer, chinese_model, output_sp_dir, output_hf_dir):
    llama_spm = pb2_model.ModelProto()
    llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())

    chinese_spm = pb2_model.ModelProto()
    chinese_spm.ParseFromString(chinese_model.serialized_model_proto())

    llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)
    print("LlamaTokenizer 中的 token 数量:", len(llama_spm_tokens_set))
    print("添加前的 token 数量:", len(llama_spm_tokens_set))

    for p in chinese_spm.pieces:
        piece = p.piece
        if piece not in llama_spm_tokens_set:
            new_p = pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            llama_spm.pieces.append(new_p)

    print("添加后的 token 数量:", len(llama_spm.pieces))

    os.makedirs(output_sp_dir, exist_ok=True)

    with open(output_sp_dir + '/Literary-alpaca2.model', 'wb') as f:
        f.write(llama_spm.SerializeToString())

    tokenizer = LlamaTokenizer(vocab_file=output_sp_dir + '/Literary-alpaca2.model')
    tokenizer.save_pretrained(output_hf_dir)
    print("Literary-alpaca2 tokenizer 已保存至:", output_hf_dir)


llama_tokenizer, chinese_model = load_tokenizers(llama_tokenizer_dir, chinese_model_file)
print_token_info(llama_tokenizer)
print_token_info(chinese_model)

save_merged_tokenizer(llama_tokenizer, chinese_model, output_sp_dir, output_hf_dir)
