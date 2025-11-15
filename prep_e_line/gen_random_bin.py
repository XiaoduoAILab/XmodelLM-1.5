import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from data_utils import create_dataloaders


# print(tokenizer)

def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default='/data2/liuyang/e_line_v6_32000/', help="data path")
    parser.add_argument("--output_path", type=str, default='/data2/liuyang/e_line_randomized.bin', help="output path")
    return parser


if __name__ == "__main__":
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    micro_batch_size = 30
    max_length = 4096
    sampling = 'random'
    # data_path = '/data2/data_e_line/'
    data_path = args.data_path
    total_batches = 500000 * 4 * 30 * 8 // micro_batch_size
    arr_len = total_batches * max_length * micro_batch_size

    data_config = [
        ("arxiv", 0.0160),
        ("book", 0.0300),
        ("c4", 0.1000),
        ("cc", 0.5451),
        ("github", 0.0150),
        ("starcoder", 0.0536),
        ("stackexchange", 0.0174),
        ("wiki", 0.0520),

        ("amps", 0.0003),
        ("Bookcorpus", 0.0006),
        ("fanfics", 0.0020),
        ("openwebmath", 0.0080),
        ("pubmed", 0.0100),

        # 中文
        ("law", 0.0100),
        ("news", 0.0050),
        ("patent", 0.0050),
        ("webtext", 0.0644),
        ("PTD", 0.0656),
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

    data_path = args.output_path
    data = np.memmap(data_path, dtype=np.uint16, mode='w+', shape=(arr_len,))

    for idx in tqdm(range(total_batches), desc=f'writing {data_path}'):
        batch = next(dataloader_iter)
        chunk_size = max_length * micro_batch_size
        data[idx * chunk_size: (idx + 1) * chunk_size] = torch.flatten(batch).cpu()

        if idx % 1024 == 0:
            data.flush()

    data.flush()
