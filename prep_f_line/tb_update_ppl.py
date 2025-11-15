import argparse
import json
from math import exp

from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default='xl_f_line', help="run name")
    parser.add_argument("--language", type=str, default='ar')
    args = parser.parse_args()

    name_maps = {'ar': 'Arabic', 'fr': 'French', 'th': 'Thai'}
    lang_name = name_maps[args.language]

    tokens_per_iter = 3440640
    log_dir = 'runs/' + args.run_name
    writer = SummaryWriter(log_dir)

    filename = f'{args.language}.jsonl'
    with open(filename, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        item = json.loads(line)
        ckp_name = item['ckp_name']
        num_iter = int(ckp_name[5:])
        num_tokens = num_iter * tokens_per_iter
        loss = item['loss']

        writer.add_scalar(f"PPL/{lang_name}", exp(loss), num_tokens)

        print(f'{num_tokens}\t{loss}')
