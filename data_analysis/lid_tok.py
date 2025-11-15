import os
import numpy as np
import json
import time
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
import sys  
sys.path.append('/data/wangyichuan')
from analysis.tokenizer import count_seq_len_sp
from analysis.lid import get_lang


data_path = '/data2/liuyang/redpajama-data-1T/arxiv'
files = os.listdir(data_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('--begin', type=int, default='0', help='')
    parser.add_argument('--end', type=int, default='5', help='')
    parser.add_argument('--interval', type=int, default='1000', help='')
    parser.add_argument('--result_path', type=str, default='/data/wangyichuan/data_analysis/arxiv/lang_tok/', help='')
    args = parser.parse_args()

    begin, end, interval = args.begin, args.end, args.interval
    result_path = args.result_path
    
    lang_ins, lang_tok = dict(), dict()
    start = time.time()
    index = 0
    for i in range(begin, end):
        with open(os.path.join(data_path, files[i]), 'r', encoding='utf-8') as f:
            for line in f:
                index += 1
                if index % interval == 0:
                    print(index, (time.time()-start)/60)
                text = json.loads(line)['text']
                lang, tok_len = get_lang(text), count_seq_len_sp(text)
                if lang not in lang_tok.keys():
                    lang_ins[lang], lang_tok[lang] = 0, 0
                lang_ins[lang] += 1
                lang_tok[lang] += tok_len


    with open(result_path+'lang_tok_%d_%d.txt'%(begin, end), 'w') as f: 
        for lang in lang_tok.keys():
            f.write('%s: \t\t%s \t%s \n'%(lang, lang_ins[lang], lang_tok[lang]))
    
    print('finish')