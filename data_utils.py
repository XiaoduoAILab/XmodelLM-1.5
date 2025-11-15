import glob
import json
import os
from typing import Tuple, Optional

from torch.utils.data import DataLoader


def load_json(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data, file_path: str):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


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
    if sampling == 'random':
        from lit_llama.packed_dataset_random import PackedDataset, CombinedDataset
    else:
        from lit_llama.packed_dataset import PackedDataset, CombinedDataset

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

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)
    return DataLoader(combined_dataset, batch_size=None, shuffle=False, pin_memory=True, num_workers=2,
                      prefetch_factor=2)


def create_dataloaders(
        batch_size: int,
        block_size: int,
        data_config,
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


