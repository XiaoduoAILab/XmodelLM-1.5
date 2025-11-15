import os
import random
import sys
from pathlib import Path

import numpy as np

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def process_bins(prefixes, ratio):
    for pre in prefixes:
        files = [f for f in os.listdir(data_path) if f.startswith(pre)]

        for f in files:
            print(f)
            filename = os.path.join(data_path, f)
            data = np.memmap(filename, dtype=np.uint16, mode='r')
            arr_len = int(round(len(data) * ratio))
            ix = random.randint(0, len(data) - arr_len - 1)
            target_fn = os.path.join(output_path, f.replace('.bin', '_dehydrated.bin'))
            target = np.memmap(target_fn, dtype=np.uint16, mode='w+', shape=(arr_len,))
            target[:] = data[ix: ix + arr_len]
            target.flush()


if __name__ == "__main__":
    data_path = '/data2/data_e_line/'
    output_path = '/home/liuyang/ds_bins/'
    prefix_en = ['arxiv', 'book', 'c4', 'cc', 'github', 'stackexchange', 'wiki', 'amps', 'Bookcorpus', 'fanfics',
                 'openwebmath', 'pubmed', 'starcoder']
    prefix_zh = ['law', 'news', 'patent', 'webtext', 'PTD']

    process_bins(prefix_en, ratio=0.511263783975492)
    process_bins(prefix_zh, ratio=0.6428305113073887)
