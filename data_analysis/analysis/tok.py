import sentencepiece as spm
from tqdm import tqdm
import numpy as np


data_path = '/data/wangyichuan/deduplicate-text-datasets/owt_50/data.txt'
bin_path = '/data/wangyichuan/deduplicate-text-datasets/owt_50/train.bin'

sp = spm.SentencePieceProcessor()  
sp.Load('/data/wangyichuan/llm_pretrain_xmodel/tokenizers/llama/tokenizer.model')


if __name__ == '__main__':
    index = 0
    with open(data_path, 'r', encoding='ISO-8859-1') as f:
        text = f.readlines()
        token = []
        for i in tqdm(range(len(text) // 2)):
            sent = text[2*i] + text[2*i+1]
            token.extend(sp.EncodeAsIds(sent))
    token = np.array(token, dtype=np.uint16)
    token.tofile(bin_path)