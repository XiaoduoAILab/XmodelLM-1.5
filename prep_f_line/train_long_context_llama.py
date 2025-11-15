import os
import pathlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, AutoModelForCausalLM

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from data_utils import create_dataloaders
from utils import reset_rope


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m"),
    tokenizer_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    template: str = field(default="plan_b")


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_args, training_args):
        super(SupervisedDataset, self).__init__()

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

        rank = int(os.environ["LOCAL_RANK"])
        seed_offset = 42 + rank

        train_dataloader, val_dataloader = create_dataloaders(
            batch_size=1,
            block_size=training_args.model_max_length,
            data_config=data_config,
            sampling="random",
            train_data_dir=data_args.data_path,
            val_data_dir=None,
            seed=1337 + seed_offset,
            vocab_size=32000
        )
        self.dataloader_iter = iter(train_dataloader)

    def __len__(self):
        return 2 ** 32

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        batch = next(self.dataloader_iter)
        # print('batch.shape: ' + str(batch.shape))
        batch = batch[0]
        return dict(input_ids=batch, labels=batch)


def make_supervised_data_module(data_args, training_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(data_args=data_args, training_args=training_args)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=None)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    rank = int(os.environ["LOCAL_RANK"])

    # `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, attn_implementation="flash_attention_2",
                                                 torch_dtype=torch.bfloat16, use_cache=False)
    model.to(f'cuda:{rank}')

    if rank == 0:
        print(model)

    # scaling_factor = 10  # hardcode here
    # reset_rope(model, model_max_train_len=81920, scaling_factor=scaling_factor)

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
