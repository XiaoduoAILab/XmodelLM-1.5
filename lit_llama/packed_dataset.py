# Very loosely inspired by indexed_dataset in Fairseq, Megatron
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/indexed_dataset.py


import random
import struct

import numpy as np
from torch.utils.data import IterableDataset

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16,
}


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24  # bytes


class PackedDataset(IterableDataset):
    def __init__(self, filenames, prefix, batch_size, block_size, seed=12345, shuffle=True):
        self._filenames = filenames
        self._prefix = prefix
        self._batch_size = batch_size
        self._block_size = block_size
        self._seed = seed
        self._shuffle = shuffle

    def __iter__(self):
        return PackedDatasetIterator(
            filenames=self._filenames,
            prefix=self._prefix,
            batch_size=self._batch_size,
            block_size=self._block_size,
            seed=self._seed,
            shuffle=self._shuffle,
        )


class PackedDatasetIterator:
    def __init__(self, filenames, prefix, batch_size, block_size, seed, shuffle):
        self._seed = seed
        self._prefix = prefix
        self._shuffle = shuffle
        self._filenames = filenames
        self._num_files = len(filenames)
        self._batch_size = batch_size
        self._block_size = block_size
        self._mmap = None
        self._file_idx = 0
        self._token_idx = 0

        self._mmap = np.memmap(self._filenames[0], mode="r", dtype=np.uint16, offset=HDR_SIZE, order="C")

    def _read_header(self, path):
        with open(path, "rb") as f:
            magic = f.read(len(HDR_MAGIC))
            assert magic == HDR_MAGIC, "File doesn't match expected format."
            version = struct.unpack("<Q", f.read(8))
            assert (1,) == version
            (dtype_code,) = struct.unpack("<B", f.read(1))
            dtype = dtypes[dtype_code]
            (chunk_size,) = struct.unpack("<Q", f.read(8))
        return dtype, chunk_size

    def _close_mmap(self):
        if self._mmap is not None:
            self._mmap._mmap.close()

    def __del__(self):
        self._close_mmap()
        del self._mmap

    def __iter__(self):
        return self

    def __next__(self):
        if self._token_idx + self._block_size * self._batch_size > len(self._mmap):
            self._file_idx = (self._file_idx + 1) % self._num_files
            self._token_idx = 0
            self._close_mmap()
            self._mmap = np.memmap(self._filenames[self._file_idx], mode="r", dtype=np.uint16, offset=HDR_SIZE,
                                   order="C")

        x = self._mmap[self._token_idx:self._token_idx + self._block_size * self._batch_size].astype(np.int64)
        x = np.reshape(x, (self._batch_size, self._block_size))
        self._token_idx += self._block_size * self._batch_size
        return x

    @property
    def prefix(self):
        return self._prefix


class CombinedDataset(IterableDataset):
    def __init__(self, datasets, seed, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)


class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        dataset, = self._rng.choices(self._datasets, weights=self._weights, k=1)
        # print(dataset.prefix)
        return next(dataset)
