import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from data_utils import create_dataloaders

if __name__ == "__main__":

    dataset = 'redpajama'
    micro_batch_size = 4 * 30 * 8 * 100
    max_length = 4096
    sampling = 'random'
    data_path = '/data3/liuyang/train_bins/'
    total_batches = 350000 * 4 * 30 * 8 // micro_batch_size
    arr_len = total_batches * max_length * micro_batch_size

    # Data proportions from https://arxiv.org/pdf/2302.13971.pdf Table 1
    data_config = [
        ("arxiv", 2.5),
        ("book", 4.5),
        ("c4", 15.0),
        ("v2", 67.0),
        ("github", 4.5),
        ("stackexchange", 2.0),
        ("wiki", 4.5),
    ]

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=max_length,
        data_config=data_config,
        sampling=sampling,
        train_data_dir=data_path,
        val_data_dir=None,
        seed=1337
    )
    dataloader_iter = iter(train_dataloader)

    data_path = '/data3/liuyang/d_line.bin'
    data = np.memmap(data_path, dtype=np.uint16, mode='w+', shape=(arr_len,))

    for idx in tqdm(range(total_batches), desc=f'writing {data_path}'):
        batch = next(dataloader_iter)
        chunk_size = max_length * micro_batch_size
        data[idx * chunk_size: (idx + 1) * chunk_size] = torch.flatten(batch).cpu()

    data.flush()
