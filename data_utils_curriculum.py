import glob
import os
import random
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


class CurriculumDataset(IterableDataset):
    def __init__(self, batch_size, block_size, data_dir, data_config, weights, shuffle, seed, vocab_size):
        self._batch_size = batch_size
        self._block_size = block_size
        self._data_dir = data_dir
        self._data_config = data_config
        self._weights = weights
        self._seed = seed
        self._shuffle = shuffle
        self._vocab_size = vocab_size

    def __iter__(self):
        return CurriculumDatasetIterator(
            batch_size=self._batch_size,
            block_size=self._block_size,
            data_dir=self._data_dir,
            data_config=self._data_config,
            weights=self._weights,
            shuffle=self._shuffle,
            seed=self._seed,
            vocab_size=self._vocab_size

        )


class CurriculumDatasetIterator:
    def __init__(self, batch_size, block_size, data_dir, data_config, weights, shuffle, seed, vocab_size):
        self._batch_size = batch_size
        self._block_size = block_size
        self._data_dir = data_dir
        self._data_config = data_config
        self._weights = weights
        self._seed = seed
        self._seed = seed
        self._shuffle = shuffle

        self._rng = random.Random(seed)

        datasets = []
        for i, (prefix, _) in enumerate(data_config):
            weight = weights[i]
            filenames = glob.glob(os.path.join(data_dir, prefix + "*.bin"))
            item = {"prefix": prefix, "filenames": filenames, "weight": weight}
            datasets.append(item)
            # print(item)

        self._datasets = datasets

        self._mmaps = dict()

        if vocab_size >= 2 ** 16:
            self._dtype = np.uint32
        else:
            self._dtype = np.uint16

    def _close_mmap(self):
        for mmap in self._mmaps.values():
            mmap._mmap.close()

    def __del__(self):
        self._close_mmap()
        del self._mmaps

    def __iter__(self):
        return self

    def __next__(self):
        x = torch.empty(size=(self._batch_size, self._block_size), dtype=torch.int64)

        for i in range(self._batch_size):
            dataset = self._rng.choices(self._datasets, weights=self._weights, k=1)[0]
            # print(dataset)
            filename = self._rng.choices(dataset['filenames'], k=1)[0]
            # print(filename)

            if filename not in self._mmaps:
                mmap = np.memmap(filename, mode="r", dtype=self._dtype, offset=0, order="C")
                self._mmaps[filename] = mmap
            else:
                mmap = self._mmaps[filename]

            ix = random.randint(0, len(mmap) - self._block_size - 1)
            x[i] = torch.from_numpy((mmap[ix:ix + self._block_size]).astype(np.int64))

        # pin arrays x, which allows us to move them to GPU asynchronously (non_blocking=True)
        # x = x.pin_memory().to('cuda', non_blocking=True)
        return x


def create_dataloader(
        batch_size: int,
        block_size: int,
        data_dir: str,
        data_config,
        sampling,
        shuffle: bool = True,
        seed: int = 12345,
        vocab_size: int = 32000
) -> DataLoader:
    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    curriculum_dataset = CurriculumDataset(batch_size=batch_size,
                                           block_size=block_size,
                                           data_dir=data_dir,
                                           data_config=data_config,
                                           weights=weights,
                                           shuffle=shuffle,
                                           seed=seed,
                                           vocab_size=vocab_size)
    return DataLoader(curriculum_dataset,
                      batch_size=None,
                      shuffle=False,
                      pin_memory=True,
                      num_workers=2,
                      prefetch_factor=2
                      )


def create_dataloaders(
        batch_size: int,
        block_size: int,
        data_config,
        sampling,
        train_data_dir: str = "/data2/wangqun/f_line_data_v11_65280/",
        val_data_dir: Optional[str] = None,
        seed: int = 12345,
        vocab_size: int = 32000
) -> Tuple[DataLoader, DataLoader]:
    """
    data_config descript the data source and data sample ratio e.g. ("wikipedia", 4.5), ("arxiv", 2.5),
     """
    # Increase by one because we need the next word as well
    # effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=block_size,
        data_dir=train_data_dir,
        data_config=data_config,
        shuffle=True,
        seed=seed,
        sampling=sampling,
        vocab_size=vocab_size
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=block_size,
            data_dir=val_data_dir,
            data_config=data_config,
            shuffle=False,
            seed=seed,
            sampling=sampling,
            vocab_size=vocab_size
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    data_config = [
        # 英文 (0-12)
        ("arxiv", 0.0151),
        ("book", 0.0282),
        ("c4", 0.0941),
        ("cc", 0.5131),
        ("github", 0.0141),
        ("starcoder", 0.0504),
        ("stackexchange", 0.0164),
        ("wiki", 0.0489),

        ("AMPS", 0.0003),
        ("BookCorpus", 0.0006),
        ("fanfics", 0.0019),
        ("Openwebmath", 0.0075),
        ("PubMed", 0.0094),

        # 中文 (13-18)
        ("law", 0.0067),
        ("news", 0.0033),
        ("patent", 0.0033),
        ("webtext", 0.0430),
        ("PTD", 0.0400),
        ("ZWJCYLK", 0.0037),

        # 小语种 (19-20)
        ("culturax", 0.0400),
        ("multilang_wiki", 0.0100),
        ("ar", 0.0167),
        ("fr", 0.0166),
        ("th", 0.0167),

    ]

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=3,
        block_size=4096,
        data_config=data_config,
        sampling='random',
        train_data_dir='/data2/wangqun/f_line_data_v11_65280/',
        val_data_dir=None,
        seed=42,
        vocab_size=65280
    )

    # training loop
    dataloader_iter = iter(train_dataloader)
    batch = next(dataloader_iter)
    print(batch)
