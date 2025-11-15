import os
import json
import time
from transformers import AutoTokenizer
import sentencepiece as spm


num = 1000000
data_path = '/data2/liuyang/redpajama-data-1T/c4/c4-train.00000-of-01024.jsonl'
# data_path = '/data2/liuyang/redpajama-data-1T/book/book.jsonl'


tokenizer = AutoTokenizer.from_pretrained('/data/wangyichuan/llm_pretrain_xmodel/tokenizers/llama', skip_special_tokens=True, add_bos_token=False,
                                          add_eos_token=False, clean_up_tokenization_spaces=True, use_fast=False, trust_remote_code=True)
def count_seq_len(text, tokenizer=tokenizer):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    tok_len = input_ids.size(1)
    return tok_len

sp = spm.SentencePieceProcessor()  
sp.Load('/data/wangyichuan/llm_pretrain_xmodel/tokenizers/llama/tokenizer.model')
def count_seq_len_sp(text, sp=sp):
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
    print(end-start)