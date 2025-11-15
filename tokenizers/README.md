# Tokenizers

## SentencePiece

训练 SentencePiece 分词器

```
$ spm_train --input=train.txt --model_prefix=xmodel --vocab_size=32000 --character_coverage=0.9999 --model_type=bpe
```

测试 SentencePiece 分词器

```
$ python experiments/tokenize_corpus.py --data_path test.txt --tokenizer tokenizers/xmodel
```

## BytePiece

训练 BytePiece 分词器

```
$ python experiments/train_bytepiece.py
```

将 BytePiece 分词器转化到与 SentencePiece 兼容

```
$ python experiments/convert_tokenizer.py
```

测试 BytePiece 分词器

```
$ python experiments/tokenize_corpus.py --data_path test.txt --tokenizer tokenizers/xmodel2
```

## 其它

使用以下命令来测试其他分词器

```
$ python experiments/tokenize_corpus.py --data_path test.txt --tokenizer tokenizers/baichuan1
```

输出如下所示：

```
BaiChuanTokenizer(name_or_path='tokenizers/baichuan1', vocab_size=64000, model_max_length=1000000000000000019884624838656, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>'}, clean_up_tokenization_spaces=False),  added_tokens_decoder={
        0: AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
        1: AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
        2: AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
}

12585648it [27:13, 7704.53it/s]
seq len: 686661289 tokenizers/baichuan1 tok len:513334321 ratio:0.7475801085970349
```