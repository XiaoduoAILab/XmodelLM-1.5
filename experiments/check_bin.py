import argparse
import os
import random

import numpy as np
import transformers


def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()

    parser.add_argument("--bin_name", type=str, default='culturax.bin')
    parser.add_argument("--tokenizer_version", type=str, default='v11', help="tokenizer version")
    parser.add_argument("--vocab_size", type=int, default=65280, help="vocab size")
    parser.add_argument("--seq_len", type=int, default=2048, help="vocab size")
    return parser


if __name__ == "__main__":
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

    data_path = '/data2/wangqun/f_line_data_v11_65280/'
    seq_len = args.seq_len
    if args.vocab_size >= 2 ** 16:
        dtype = np.uint32
    else:
        dtype = np.uint16
    data = np.memmap(os.path.join(data_path, args.bin_name), dtype=dtype, mode='r')
    ix = random.randrange(start=0, stop=len(data) - seq_len)

    seq = data[ix:ix + seq_len]
    text = tokenizer.decode(seq)
    print(text)
