import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class RedpajamaDataset(Dataset):
    """Redpajama training dataset."""

    def __init__(self, data_dir, max_length, batch_size, sampling='random'):
        self.data_dir = data_dir
        self.max_length = max_length
        self.batch_size = batch_size
        shuffle = True
        seed_offset = torch.distributed.get_rank()
        seed = 1337 + seed_offset

        if sampling == 'random':
            from lit_llama.packed_dataset_random_nssc import PackedDataset, CombinedDataset
        else:
            from lit_llama.packed_dataset_nssc import PackedDataset, CombinedDataset

        # Data proportions from table 1, https://arxiv.org/pdf/2302.13971.pdf
        data_config = [
            ("arxiv", 2.5),
            ("book", 4.5),
            ("c4", 15.0),
            ("cc", 67.0),
            ("github", 4.5),
            ("stackexchange", 2.0),
            ("wikipedia", 4.5),
        ]
        # data_config = [("train", 100)]

        datasets = []
        for prefix, _ in data_config:
            filenames = glob.glob(os.path.join(data_dir, prefix + "*"))
            dataset = PackedDataset(filenames, prefix, batch_size=batch_size, block_size=max_length, shuffle=shuffle,
                                    seed=seed)
            datasets.append(dataset)

        if not datasets:
            raise RuntimeError(
                f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
            )

        weights = [weight for _, weight in data_config]
        sum_weights = sum(weights)
        weights = [el / sum_weights for el in weights]

        self.data_iter = iter(CombinedDataset(datasets=datasets, seed=seed, weights=weights))

    def __len__(self):
        return 2 ** 32  # big integer!

    def __getitem__(self, idx):
        return next(self.data_iter)


class OpenWebTextDataset(Dataset):
    """OpenWebText validation dataset."""

    def __init__(self, split, max_length, batch_size):
        self.max_length = max_length
        self.batch_size = batch_size
        self.data = np.memmap(os.path.join('data/openwebtext', f'{split}.bin'), dtype=np.uint16, mode='r')
        self.data_length = len(self.data)

    def __len__(self):
        return 2 ** 32  # big integer!

    def __getitem__(self, idx):
        ix = torch.randint(self.data_length - self.max_length - 1, (1,))[0]
        # batch = torch.stack([torch.from_numpy((self.data[i:i + self.max_length]).astype(np.int64)) for i in ix])
        batch = torch.from_numpy((self.data[ix:ix + self.max_length + 1]).astype(np.int64))
        # pin arrays batch, which allows us to move them to GPU asynchronously (non_blocking=True)
        # batch = batch.pin_memory().to(device, non_blocking=True)
        sample_dict = {'text': np.array(batch, dtype=np.int64)}
        return sample_dict
