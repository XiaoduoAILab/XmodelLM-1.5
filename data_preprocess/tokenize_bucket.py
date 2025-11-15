import os
import time
import sentencepiece as spm
from tqdm import tqdm
import numpy as np
import argparse


data_dir = '/data1/redpajama-v2-process/data_lsh_2'
# bin_dir = '/data1/redpajama-v2-process/bins_xmodel'
# bin_dir = '/data1/redpajama-v2-process/bins_xmodel_64000'
# bin_dir = '/data1/redpajama-v2-process/bins_xmodel_v6_64000'
# bin_dir = '/data1/redpajama-v2-process/bins_xmodel_v6_32000'
bin_dir = '/data1/redpajama-v2-process/bins_xmodel_v11_65280'

sp = spm.SentencePieceProcessor()
# sp.Load('/data1/llm_pretrain_xmodel/tokenizers/llama/tokenizer.model')
# sp.Load('/data1/redpajama-v2-process/llm_pretrain_xmodel/tokenizers/xmodel/xmodel_32000.model')
# sp.Load('/data1/redpajama-v2-process/llm_pretrain_xmodel/tokenizers/xmodel_64000/xmodel_64000.model')
# sp.Load('/data1/redpajama-v2-process/llm_pretrain_xmodel/tokenizers/xmodel/v6/xmodel_64000/xmodel_64000.model')
# sp.Load('/data1/redpajama-v2-process/llm_pretrain_xmodel/tokenizers/xmodel/v6/xmodel_32000/xmodel_32000.model')
sp.Load('/data2/liuyang/llm_pretrain_xmodel/tokenizers/xmodel/v11/xmodel_65280/xmodel_65280.model')



batch_size = 10000    # 1.1B lines

def generate_lines(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield line


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='2014-15', help='')
    parser.add_argument('--add_eos', type=bool, default=True, help='')
    parser.add_argument('--start', type=int, default=0, help='')
    parser.add_argument('--end', type=int, default=10000, help='')
    args = parser.parse_args()
    file, start, end, add_eos = args.file, args.start, args.end, args.add_eos

    data_path, bin_path = os.path.join(data_dir, file+'.txt'), os.path.join(bin_dir, file+'_'+str(start)+'.bin')
    index, num, flag = 0, 0, time.time()
    with open(bin_path, 'a') as f_bin:
        n = 0
        lines_generator = generate_lines(data_path)
        lines, batch_lines = '', []
        for line in lines_generator:
            if line == '[end of paragraph --wyc]\n':
                if n == end:
                    break
                n += 1
                if n <= start:
                    lines = ''
                    if n % 1000000 == 0:
                        print(n, round((time.time()-flag)/60, 2))
                    continue
                batch_lines.append(lines)
                lines = ''
            else:
                lines += line
            if len(batch_lines) == batch_size:
                num += 1
                print('batch read: %d time: %.4f'%(num, (time.time()-flag)/60))
                if add_eos:
                    tokens = []
                    for line in batch_lines:
                        tokens.extend(sp.EncodeAsIds(line) + [2])
                else:
                    tokens = []
                    for line in batch_lines:
                        tokens.extend(sp.EncodeAsIds(line))
                token_array = np.array(tokens, dtype=np.uint16)
                token_array.tofile(f_bin)
                batch_lines = []
                print('batch saved: %d time: %.4f'%(num, (time.time()-flag)/60))
        
        if batch_lines:
            if add_eos:
                tokens = []
                for line in batch_lines:
                    tokens.extend(sp.EncodeAsIds(line) + [2])
            else:
                tokens = []
                for line in batch_lines:
                    tokens.extend(sp.EncodeAsIds(line))
            token_array = np.array(tokens, dtype=np.uint16)
            token_array.tofile(f_bin)
    
    print('finish')