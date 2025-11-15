import json
import time

import sentencepiece as spm
from transformers import AutoTokenizer

num = 1000000
data_path = '/data/wangyichuan/deduplicate-text-datasets/owt_50/data.txt'


# data_path = '/data2/liuyang/redpajama-data-1T/book/book.jsonl'


def count_seq_len(text, tokenizer=None):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained('/data/wangyichuan/llm_pretrain_xmodel/tokenizers/llama',
                                                  skip_special_tokens=True, add_bos_token=False,
                                                  add_eos_token=False, clean_up_tokenization_spaces=True,
                                                  use_fast=False, trust_remote_code=True)
    input_ids = tokenizer.encode(text, return_tensors="pt")
    tok_len = input_ids.size(1)
    return tok_len


def count_seq_len_sp(text, sp=None):
    if sp is None:
        sp = spm.SentencePieceProcessor()
        sp.Load('/data/wangyichuan/llm_pretrain_xmodel/tokenizers/llama/tokenizer.model')
    input_ids = sp.EncodeAsPieces(text)
    return len(input_ids)


if __name__ == '__main__':
    start = time.time()
    index = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = json.loads(line)['text']
            tok = count_seq_len_sp(text)
            index += 1
            if index == num:
                break
    end = time.time()
    print(end - start)
