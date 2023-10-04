from transformers import LlamaTokenizer

hf_dir = './incorporation_hf'
Alpaca_dir = './Chinese-Alpaca-tokenizer'

text = """
第一式龙争虎斗确实钢猛，似乎将一身之力灌注于刀身。这一刀下去开山裂石可能差一些，不过要把白云给劈成两半那是绰绰有余。
　　眼看这一刀到了眼前，白云伸双手握住三叉戟接这一招。因为力道非常强大，所以就连白云也不敢怠慢。
　　刀戟互碰之间，白云惊出了一身的冷汗。看似钢猛的一刀却毫无力度，就像是刀口轻轻的放到了戟身，就连互碰都称不上。但此时想要换招却并不容易。为了接王富贵这一招可是用了五成力道伸了双手往上顶，想不到这一下扑了个空，身体一下就不受控制往前倾。
　　王富贵把握机会使出了第二式，剃鳞，第三式，卸甲。"""



def test_tokenizers(tokenizer_dir, text):
    llama_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_dir)

    return llama_tokenizer.tokenize(text)


print("Literary-alpaca2 tokenizer 分词结果:", test_tokenizers(hf_dir,text))
print("Alpaca tokenizer 分词结果:",test_tokenizers(Alpaca_dir,text))
