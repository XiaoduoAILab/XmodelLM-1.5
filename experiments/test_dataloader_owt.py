import os

import numpy as np
import torch
from tqdm import tqdm


def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])

    # pin arrays x, which allows us to move them to GPU asynchronously (non_blocking=True)
    x = x.pin_memory().to(device, non_blocking=True)
    return x


def get_batch_in_order(data):
    global ix
    # ix = torch.randint(len(data) - block_size, (batch_size,))
    # x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    x = data[ix:ix + block_size * batch_size].astype(np.int64)
    x = torch.from_numpy(x)
    x = torch.reshape(x, (batch_size, block_size))
    # pin arrays x, which allows us to move them to GPU asynchronously (non_blocking=True)
    x = x.pin_memory().to(device, non_blocking=True)
    ix += block_size * batch_size
    return x


if __name__ == "__main__":
    batch_size = 10
    block_size = 4096
    total_steps = 100
    device = 'cuda'
    in_order = False

    dataset = 'openwebtext'
    data_dir = os.path.join('data', dataset)
    filename = os.path.join(data_dir, 'train.bin')
    ix = 0
    print('filename: ' + str(filename))
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    print('data: ' + str(data))
    print('len(data): ' + str(len(data)))

    for _ in tqdm(range(total_steps)):
        if in_order:
            get_batch_in_order(data)
        else:
            get_batch(data)

    HDR_SIZE = 24
    filename = 'book_0000000000.bin'
    ix = 0
    print('filename: ' + str(filename))
    data = np.memmap(filename, dtype=np.uint16, mode='r', offset=HDR_SIZE, order="C")
    # data = np.memmap(filename, dtype=np.uint16, mode='r', order="C")
    print('data: ' + str(data))
    print('len(data): ' + str(len(data)))

    for _ in tqdm(range(total_steps)):
        if in_order:
            get_batch_in_order(data)
        else:
            get_batch(data)

    HDR_SIZE = 24
    filename = '/data1/liuyang/lit-redpajama-new/book_0000000000.bin'
    ix = 0
    print('filename: ' + str(filename))
    data = np.memmap(filename, dtype=np.uint16, mode='r', offset=HDR_SIZE, order="C")
    # data = np.memmap(filename, dtype=np.uint16, mode='r', order="C")
    print('data: ' + str(data))
    print('len(data): ' + str(len(data)))

    for _ in tqdm(range(total_steps)):
        if in_order:
            get_batch_in_order(data)
        else:
            get_batch(data)

    filename = 'book.bin'
    ix = 0
    print('filename: ' + str(filename))
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    print('data: ' + str(data))
    print('len(data): ' + str(len(data)))

    for _ in tqdm(range(total_steps)):
        if in_order:
            get_batch_in_order(data)
        else:
            get_batch(data)
