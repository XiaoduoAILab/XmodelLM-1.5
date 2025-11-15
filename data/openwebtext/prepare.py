# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import argparse
import os

import numpy as np
import transformers
from datasets import load_dataset  # huggingface datasets
from tqdm import tqdm


def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=r'C:\datasets\openwebtext.tar.xz', help="data path")
    parser.add_argument("--tokenizer_version", type=str, default='v11', help="tokenizer version")
    parser.add_argument("--vocab_size", type=int, default=65280, help="vocab size")
    return parser


if __name__ == "__main__":
    num_proc = 16
    num_proc_load_dataset = num_proc

    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    data_path = args.data_path

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        f'tokenizers/xmodel/{args.tokenizer_version}/xmodel_{args.vocab_size}',
        skip_special_tokens=True,
        add_bos_token=False,
        add_eos_token=False,
        clean_up_tokenization_spaces=True,
        use_fast=False,
        trust_remote_code=True)

    # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
    dataset = load_dataset("openwebtext", data_files=data_path)

    # owt by default only contains the 'train' split, so create a test split
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # rename the test split to val
    split_dataset.pop('train')


    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 8009762
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 4007
    #     })
    # })

    def process(example, tokenizer):
        ids = tokenizer.encode(example['text'])  # encode_ordinary ignores any special tokens
        eos_token = 2
        ids.append(eos_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out


    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
        fn_kwargs={"tokenizer": tokenizer}
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}_{args.tokenizer_version}_{args.vocab_size}.bin')
        if args.vocab_size >= 2 ** 16:
            dtype = np.uint32
        else:
            dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # train.bin is ~17GB, val.bin ~8.5MB
    # train has ~9B tokens (9,035,582,198)
    # val has ~4M tokens (4,434,897)

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
