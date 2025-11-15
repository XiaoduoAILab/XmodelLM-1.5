import os
import json
import argparse
from tqdm import tqdm
from gensim.utils import simple_preprocess  
from simhash import Simhash, SimhashIndex


threshold = 3
f = 128
bin_size = 10000000


def lsh_bucket(data: list, output_path, f=f, threshold=threshold, bin_size=bin_size):
    bins = list(range(0, len(data), bin_size)) + [len(data)]
    num_bins = len(bins) - 1
    dele = []
    for i in range(num_bins):
        index = SimhashIndex([], k=threshold, f=f)
        temp_dele = []
        for i in tqdm(range(bins[i], bins[i+1])):
            text = data[i]
            hash = Simhash(simple_preprocess(text), f=f)
            if index.judge_dup(hash):
                temp_dele.append(i)
            else:
                index.add(i, hash)
        dele += temp_dele
    print('total: %d delete: %d save: %d'%(len(data), len(dele), len(data)-len(dele)))
    for i in dele[::-1]:
        data.pop(i)

    with open(output_path, 'a', encoding='utf-8') as f:
        for line in data:
            json.dump({'text': line}, f, ensure_ascii=False)
            f.write('\n')


def lsh(input_dir, output_path, start, end, f=f, threshold=threshold, bin_size=bin_size):
    files = sorted(os.listdir(input_dir))[start:end]
    paras = []
    for file in files:
        data_path = os.path.join(input_dir, file)
        with open(data_path, 'r', encoding='utf-8') as fp:
            for line in tqdm(fp):
                paras.append(json.loads(line)['text'])
                if len(paras) == bin_size:
                    lsh_bucket(paras, output_path, f=f, threshold=threshold, bin_size=bin_size)
                    paras = []
    lsh_bucket(paras, output_path, f=f, threshold=threshold, bin_size=bin_size)
    print('finish')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/data3/wangyichuan/domain_data/data_preprocess')
    parser.add_argument('--output_dir', type=str, default='/data3/wangyichuan/domain_data/data_dedup')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=10000)
    parser.add_argument('--src', type=str, default='ks')
    parser.add_argument('--year', type=int, default=2024)
    args = parser.parse_args()

    input_dir, output_dir, start, end, src, year = args.input_dir, args.output_dir, args.start, args.end, args.src, args.year
    input_dir = os.path.join(input_dir, str(year), src)
    output_path = os.path.join(output_dir, str(year), f'{src}_{start}_{end}.json')
    lsh(input_dir, output_path, start, end)
