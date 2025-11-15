import os
import json
import time

import numpy as np
import argparse
import sentencepiece as spm


sp = spm.SentencePieceProcessor()
# sp.Load('/data/wangyichuan/process/llm_pretrain_xmodel/tokenizers/xmodel/v6/xmodel_32000/xmodel_32000.model')
sp.Load('/data1/liuyang/llm_pretrain_xmodel/tokenizers/xmodel/v6/xmodel_32000/xmodel_32000.model')

add_eos = True
batch_size = 10000

def generate_lines(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield json.loads(line)['text']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data3/wangyichuan/domain_data/data_dedup', help='')
    parser.add_argument('--bin_dir', type=str, default='/data3/wangyichuan/domain_data/data_bins', help='')
    parser.add_argument('--file', type=str, default='test', help='')
    parser.add_argument('--start', type=int, default=0, help='')
    parser.add_argument('--end', type=int, default=1000000000, help='')
    parser.add_argument('--add_eos', type=bool, default=True, help='')
    parser.add_argument('--year', type=int, default=2024)
    args = parser.parse_args()
    data_dir, bin_dir, file, add_eos, year = args.data_dir, args.bin_dir, args.file, args.add_eos, args.year
    data_path, bin_path = os.path.join(data_dir, str(year), file+'.json'), os.path.join(bin_dir, str(year), file+'.bin')
    start, end = args.start, args.end

    n, num, flag = 0, 0, time.time()
    with open(bin_path, 'w') as f_bin:
        lines_generator = generate_lines(data_path)
        batch_lines = []
        for line in lines_generator:
            n += 1
            if n == end:
                break
            if n <= start:
                if n % 1000000 == 0:
                    print(n, round((time.time()-flag)/60, 2))
                continue
            batch_lines.append(line)

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
