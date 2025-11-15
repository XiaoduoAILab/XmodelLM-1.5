import argparse

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/data2/wangqun/f_line_data_v11_65280/ZWJCYLK.bin')
    parser.add_argument("--tokenizer_path", type=str, default='tokenizers/xmodel/v11/xmodel_65280')
    return parser


if __name__ == "__main__":
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    data = np.memmap(args.data_path, dtype=np.uint16, mode='r')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    for i in tqdm(range(0, len(data), 4096)):
        toks = data[i:i + 4096]
        text = tokenizer.decode(toks)

        keyword = '之江新语'
        if keyword in text:
            idx = text.index(keyword)

            print(text[idx - 100:idx + 100])
