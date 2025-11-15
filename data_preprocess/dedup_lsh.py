import os
import time
import argparse
from tqdm import tqdm
from gensim.utils import simple_preprocess  
from simhash import Simhash, SimhashIndex


threshold = 4
f = 128
bin_size = 200000

start, end = 0, -1
data_dir = '/data1/redpajama-v2-process/data_preprocess'
output_dir = '/data1/redpajama-v2-process/data_lsh'


def read_data(data_path, start=start, end=end):
    paras, n = [], 0
    with open(data_path, 'r', encoding='utf-8') as f:
        para = ''
        for line in tqdm(f):
            n += 1
            if n < start:
                continue
            elif n == end+1:
                break

            if line == '[end of paragraph --wyc]\n':
                para += line
                paras.append(para)
                para = ''
            else:
                para += line
    if len(para) > 1:
        paras.append(para)

    print('paras: %d'%len(paras))
    return paras


def lsh(data: list, output_path, threshold=threshold, bin_size=bin_size):
    bins = list(range(0, len(data), bin_size)) + [len(data)]
    num_bins = len(bins) - 1
    dele = []
    for i in range(num_bins):
        index = SimhashIndex([], k=threshold, f=f)
        temp_dele = []
        for i in tqdm(range(bins[i], bins[i+1])):
            text = data[i]
            hash = Simhash(simple_preprocess(text), f=f)
            # sim_set = index.get_near_dups(hash)
            # if len(sim_set) > 0:
            #     temp_dele.append(i)
            # else:
            #     index.add(i, hash)
            if index.judge_dup(hash):
                temp_dele.append(i)
            else:
                index.add(i, hash)
        dele += temp_dele
    print('total: %d  delete: %d  save: %d'%(len(data), len(dele), len(data)-len(dele)))
    for i in dele[::-1]:
        data.pop(i)
    with open(output_path, 'a') as fp:
        all = [line + '\n[end of paragraph --wyc]\n' for line in data]
        fp.writelines(all)


def lsh_data(data_path, output_path, start=start, end=end, threshold=threshold, bin_size=bin_size):
    paras, n = [], 0
    with open(data_path, 'r', encoding='utf-8') as f:
        para = ''
        for line in tqdm(f):
            n += 1
            if n < start:
                continue
            elif n == end+1:
                break

            if line == '[end of paragraph --wyc]\n':
                paras.append(para)
                para = ''
            else:
                para += line
            if len(paras) == bin_size:
                lsh(paras, output_path, threshold=threshold, bin_size=bin_size)
                paras = []
    if len(para) > 1:
        paras.append(para)
    lsh(paras, output_path, threshold=threshold, bin_size=bin_size)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='2014-15', help='')
    args = parser.parse_args()
    file = args.file

    data_path, output_path = os.path.join(data_dir, file+'.txt'), os.path.join(output_dir, file+'.txt')
    # data = read_data(data_path, start, end)
    # lsh(data, output_path)
    lsh_data(data_path, output_path)