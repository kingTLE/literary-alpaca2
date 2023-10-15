from transformers import LlamaTokenizer

hf_dir = './incorporation_hf'
Alpaca_dir = './Chinese-Alpaca-tokenizer'
llama='./llama-2-13b-hf-tokenizer'
text = """
挟太山以超北海，语人曰：‘我不能。’是诚不能也。
为长者折枝，语人曰：‘我不能。’是不为也，非不能也。
故王之不王，非挟太山以超北海之类也；王之不王，是折枝之类也。
老吾老，以及人之老；幼吾幼，以及人之幼：天下可运于掌。
《诗》云：‘刑于寡妻，至于兄弟，以御于家邦。
"""
def test_tokenizers(tokenizer_dir, text):
    llama_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_dir)

    return llama_tokenizer.tokenize(text)


print("Literary-alpaca2 tokenizer 分词结果:", test_tokenizers(hf_dir,text))
print("llama2原生 tokenizer 分词结果:", test_tokenizers(llama,text))
print("Alpaca tokenizer 分词结果:",test_tokenizers(Alpaca_dir,text))
