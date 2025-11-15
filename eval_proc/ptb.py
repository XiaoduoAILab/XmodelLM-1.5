import argparse

import torch
import transformers
from datasets import load_dataset

from utils import eval_ppl


def eval_ptb(model, tokenizer):
    data = load_dataset("text", data_files={"test": "data/ptb/ptb.test.txt"}, split='test')['text']
    # print(data)
    # print('len(data): ' + str(len(data)))
    return eval_ppl(data, model, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ptb')
    parser.add_argument('--model', type=str, default=r'D:\models\7B',
                        help='pretrained model name or path')
    parser.add_argument('--stride', type=int, default=512, help='stride')

    args = parser.parse_args()
    print(args)

    device = "cuda"

    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                              trust_remote_code=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)

    model.cuda()
    model.eval()

    print(model)
    print(tokenizer)

    ppl = eval_ptb(model, tokenizer)
    print('ppl: ' + str(ppl))
