# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import argparse
import os

import numpy as np
import transformers
from datasets import load_dataset  # huggingface datasets


# print(tokenizer)

def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer_version", type=str, default='v11', help="tokenizer version")
    parser.add_argument("--vocab_size", type=int, default=65280, help="vocab size")
    return parser


if __name__ == "__main__":
    num_proc = 8
    num_proc_load_dataset = num_proc

    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        f'tokenizers/xmodel/{args.tokenizer_version}/xmodel_{args.vocab_size}',
        skip_special_tokens=True,
        add_bos_token=False,
        add_eos_token=False,
        clean_up_tokenization_spaces=True,
        use_fast=False,
        trust_remote_code=True)

    print(tokenizer.encode('\n'))

    # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
    dataset = load_dataset("text", data_files={"test": "data/culturax_zh/culturax_zh.txt"}, split='test')['text']

    encodings = tokenizer("\n\n".join(dataset), return_tensors="pt")
    arr_len = encodings.input_ids.size(1)

    filename = os.path.join(os.path.dirname(__file__), f'culturax_zh_{args.tokenizer_version}_{args.vocab_size}.bin')
    if args.vocab_size >= 2 ** 16:
        dtype = np.uint32
    else:
        dtype = np.uint16
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 1024

    arr[:] = encodings.input_ids[:]
    arr.flush()
