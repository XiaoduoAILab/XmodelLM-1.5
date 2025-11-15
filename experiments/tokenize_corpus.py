import argparse

from tqdm import tqdm
from transformers import AutoTokenizer


def count_seq_len(tokenizer):
    tok_len = 0
    seq_len = 0
    with open(args.data_path, 'r') as fp:
        for line in tqdm(fp):
            input_ids = tokenizer.encode(line, return_tensors="pt")
            tok_len += input_ids.size(1)
            seq_len += len(line)
    return seq_len, tok_len


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model')
    # general
    parser.add_argument('--data_path', type=str, default='/data4/mishiqian/data/msg.txt', help='')
    parser.add_argument('--tokenizer', type=str, default='tokenizers/xmodel_32000', help='')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    print(tokenizer)
    # print(tokenizer.get_vocab())

    seq_len, tok_len = count_seq_len(tokenizer)
    print(f'seq len: {seq_len} {args.tokenizer} tok len:{tok_len} ratio:{tok_len / seq_len}')
