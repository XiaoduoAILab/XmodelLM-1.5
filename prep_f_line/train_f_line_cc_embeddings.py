import glob
import os
import pathlib
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import Trainer

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

IGNORE_INDEX = -100
HDR_SIZE = 24  # bytes


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default='/data1/.cache')
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


class CommonCrawlDataset(torch.utils.data.Dataset):
    def __init__(self, data_args, training_args):
        super(CommonCrawlDataset, self).__init__()

        prefix = 'cc'
        seed = 42
        self._block_size = 2048
        self._filenames = glob.glob(os.path.join(data_args.data_path, prefix + "*.bin"))
        self._mmaps = dict()
        self._rng = random.Random(seed)

    def __len__(self):
        return 128 * 50000

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        filename = self._rng.choice(self._filenames)

        if filename not in self._mmaps:
            mmap = np.memmap(filename, mode="r", dtype=np.uint16, offset=HDR_SIZE, order="C")
            self._mmaps[filename] = mmap
        else:
            mmap = self._mmaps[filename]

        ix = random.randint(0, len(mmap) - self._block_size)
        x = torch.from_numpy((mmap[ix:ix + self._block_size]).astype(np.int64))
        return dict(input_ids=x, labels=x)


def make_supervised_data_module(data_args, training_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = CommonCrawlDataset(data_args=data_args, training_args=training_args)
    return dict(train_dataset=train_dataset, eval_dataset=None)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype="auto",
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        use_cache=False
    )
    from patch.flash_attention_patch import replace_with_flash_attention
    replace_with_flash_attention(model)

    model.model.requires_grad_(False)
    model.get_input_embeddings().requires_grad_(True)
    model.get_output_embeddings().requires_grad_(True)

    # trainable_parameters = count_parameters(model)
    # print('trainable_parameters: ' + str(trainable_parameters))

    data_module = make_supervised_data_module(data_args=data_args, training_args=training_args)
    trainer = Trainer(model=model, args=training_args, **data_module)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
