import argparse

import torch
import transformers
from datasets import Dataset, concatenate_datasets

from utils import eval_ppl

if __name__ == "__main__":
    arrow_files = ["data/wikitext103/wikitext-train-00000-of-00002.arrow",
                   "data/wikitext103/wikitext-train-00001-of-00002.arrow",
                   "data/wikitext103/wikitext-validation.arrow",
                   "data/wikitext103/wikitext-test.arrow", ]

    data = concatenate_datasets([Dataset.from_file(arrow_file) for arrow_file in arrow_files])["text"]
    # print(data)
    print('len(data): ' + str(len(data)))

    parser = argparse.ArgumentParser(description='wikitext103')
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

    ppl = eval_ppl(data, model, tokenizer)
    print('ppl: ' + str(ppl))
