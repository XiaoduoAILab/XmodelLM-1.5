import argparse
import json
from math import exp

from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default='data_mix_1_to_1', help="run name")
    parser.add_argument("--mask_unrelated", action="store_true", help="mask unrelated")
    args = parser.parse_args()

    tokens_per_iter = 983040
    log_dir = 'runs/' + args.run_name
    writer = SummaryWriter(log_dir)

    filename = f'{args.run_name}.jsonl'
    with open(filename, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        item = json.loads(line)
        ckp_name = item['ckp_name']
        num_iter = int(ckp_name[5:])
        num_tokens = num_iter * tokens_per_iter
        loss = item['loss']
        if args.mask_unrelated:
            writer.add_scalar("Main/test_loss_masked", loss, num_tokens)
            writer.add_scalar(f"PPL/In_Domain_Masked", exp(loss), num_tokens)
        else:
            writer.add_scalar("Main/test_loss", loss, num_tokens)
            writer.add_scalar(f"PPL/In_Domain", exp(loss), num_tokens)

        print(f'{num_tokens}\t{loss}')
