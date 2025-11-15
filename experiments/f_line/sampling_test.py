import glob
import os
import random
from typing import Tuple, Optional

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from tqdm import tqdm


class PackedDataset(IterableDataset):
    def __init__(self, filenames, prefix, batch_size, block_size, seed=12345, shuffle=True, vocab_size=32000):
        self._filenames = filenames
        self._prefix = prefix
        self._batch_size = batch_size
        self._block_size = block_size
        self._seed = seed
        self._shuffle = shuffle
        self._vocab_size = vocab_size

    def __iter__(self):
        return PackedDatasetIterator(
            filenames=self._filenames,
            prefix=self._prefix,
            batch_size=self._batch_size,
            block_size=self._block_size,
            seed=self._seed,
            shuffle=self._shuffle,
            vocab_size=self._vocab_size
        )


class PackedDatasetIterator:
    def __init__(self, filenames, prefix, batch_size, block_size, seed, shuffle, vocab_size):
        self._seed = seed
        self._rng = random.Random(seed)
        self._prefix = prefix
        self._shuffle = shuffle
        self._vocab_size = vocab_size
        self._filenames = filenames
        self._num_files = len(filenames)
        self._batch_size = batch_size
        self._block_size = block_size
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
        filename = self._rng.choice(self._filenames)
        # print(filename)
        # pin arrays x, which allows us to move them to GPU asynchronously (non_blocking=True)
        # x = x.pin_memory().to('cuda', non_blocking=True)
        return filename

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
        global weights
        print('weights: ' + str(weights))
        print('self._datasets: ' + str(self._datasets))
        dataset, = self._rng.choices(self._datasets, weights=weights, k=1)
        # print(dataset.prefix)
        return next(dataset)


def create_dataloader_curriculum(
        batch_size: int,
        block_size: int,
        data_dir: str,
        data_config,
        weights,
        sampling,
        shuffle: bool = True,
        seed: int = 12345,
        vocab_size: int = 32000
) -> DataLoader:
    datasets = []
    for prefix, _ in data_config:
        filenames = glob.glob(os.path.join(data_dir, prefix + "*.bin"))
        # print(prefix, filenames)
        dataset = PackedDataset(filenames, prefix, batch_size=batch_size, block_size=block_size, shuffle=shuffle,
                                seed=seed, vocab_size=vocab_size)
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    # weights = [weight for _, weight in data_config]
    # sum_weights = sum(weights)
    # weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)
    return DataLoader(combined_dataset, batch_size=None, shuffle=False, pin_memory=True, num_workers=2,
                      prefetch_factor=2)


def create_dataloaders_curriculum(
        batch_size: int,
        block_size: int,
        data_config,
        weights,
        sampling,
        train_data_dir: str = "data/lit-redpajama",
        val_data_dir: Optional[str] = None,
        seed: int = 12345,
        vocab_size: int = 32000
) -> Tuple[DataLoader, DataLoader]:
    """
    data_config descript the data source and data sample ratio e.g. ("wikipedia", 4.5), ("arxiv", 2.5),
     """
    # Increase by one because we need the next word as well
    # effective_block_size = block_size + 1
    train_dataloader = create_dataloader_curriculum(
        batch_size=batch_size,
        block_size=block_size,
        data_dir=train_data_dir,
        data_config=data_config,
        weights=weights,
        shuffle=True,
        seed=seed,
        sampling=sampling,
        vocab_size=vocab_size
    )
    val_dataloader = (
        create_dataloader_curriculum(
            batch_size=batch_size,
            block_size=block_size,
            data_dir=val_data_dir,
            data_config=data_config,
            weights=weights,
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
    data_config_start = [
        # 英文 (0-12)
        ("arxiv", 0.0189),
        ("book", 0.0352),
        ("c4", 0.1176),
        ("cc", 0.6415),
        ("github", 0.0176),
        ("starcoder", 0.0630),
        ("stackexchange", 0.0205),
        ("wiki", 0.0610),

        ("AMPS", 0.0004),
        ("BookCorpus", 0.0007),
        ("fanfics", 0.0024),
        ("Openwebmath", 0.0094),
        ("PubMed", 0.0118),

        # 中文 (13-18)
        ("law", 0.0000),
        ("news", 0.0000),
        ("patent", 0.0000),
        ("webtext", 0.0000),
        ("PTD", 0.0000),
        ("ZWJCYLK", 0.0000),

        # 小语种 (19-20)
        ("culturax", 0.0000),
        ("multiwiki", 0.0000),
    ]

    weights_start = [weight for _, weight in data_config_start]
    sum_weights = sum(weights_start)
    weights_start = np.array([el / sum_weights for el in weights_start])

    data_config_end = [
        # 英文 (0-12)
        ("arxiv", 0.0160),
        ("book", 0.0300),
        ("c4", 0.1000),
        ("cc", 0.5451),
        ("github", 0.0150),
        ("starcoder", 0.0536),
        ("stackexchange", 0.0174),
        ("wiki", 0.0520),

        ("AMPS", 0.0003),
        ("BookCorpus", 0.0006),
        ("fanfics", 0.0020),
        ("Openwebmath", 0.0080),
        ("PubMed", 0.0100),

        # 中文 (13-18)
        ("law", 0.0067),
        ("news", 0.0033),
        ("patent", 0.0033),
        ("webtext", 0.0430),
        ("PTD", 0.0400),
        ("ZWJCYLK", 0.0037),

        # 小语种 (19-20)
        ("culturax", 0.0400),
        ("multiwiki", 0.0100),
    ]

    weights_end = [weight for _, weight in data_config_end]
    sum_weights = sum(weights_end)
    weights_end = np.array([el / sum_weights for el in weights_end])

    # 真正起作用的是这个
    weights = [weight for _, weight in data_config_start]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    micro_batch_size = 3
    max_length = 4096
    sampling = 'random'
    data_path = '/data2/wangqun/f_line_data_v11_65280/'
    seed_offset = 42
    vocab_size = 65280

    train_dataloader, val_dataloader = create_dataloaders_curriculum(
        batch_size=micro_batch_size,
        block_size=max_length,
        data_config=data_config_start,
        weights=weights,
        sampling=sampling,
        train_data_dir=data_path,
        val_data_dir=None,
        seed=1337 + seed_offset,
        vocab_size=vocab_size
    )

    dataloader_iter = iter(train_dataloader)

    total_steps = 600000
    start_iter = 300000
    end_iter = total_steps + 1
    stop_it = 300010
    with open('samples.txt', 'w') as fp:
        for iter_num in tqdm(range(start_iter, end_iter)):
            alpha = iter_num / total_steps
            _weights = weights_start + alpha * (weights_end - weights_start)
            _weights = _weights.tolist()

            # assert (len(_weights) == len(weights), f'{len(_weights)} is not equal to {len(weights)}')
            length = len(weights)
            for i in range(length):
                weights[i] = _weights[i]

            print(weights)

            batch = next(dataloader_iter)

            fp.write(f'{batch}\n')

            if iter_num > stop_it:
                break
