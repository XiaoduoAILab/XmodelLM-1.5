import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from tqdm import tqdm

wd = Path(__file__).parent.parent.parent.resolve()
print('wd: ' + str(wd))
sys.path.append(str(wd))

from evaluate.utils import deploy_model


def mask_unrelated(batch):
    new_batch = torch.ones(size=(batch_size, max_length), dtype=torch.long) * -100

    for i in range(batch_size):
        drawing_mode = False
        for j in range(max_length):
            if j <= max_length - 3 and batch[i, j] == 6963 and batch[i, j + 1] == 7522 and batch[i, j + 2] == 392:
                drawing_mode = True

            if drawing_mode:
                new_batch[i, j] = batch[i, j]

            if (j <= max_length - 2 and batch[i, j] == 97 and batch[i, j + 1] == 6233) or (
                    j <= max_length - 2 and batch[i, j] == 97 and batch[i, j + 1] == 7213):
                drawing_mode = False

    return new_batch


def get_batch():
    ix = torch.randint(len(data) - max_length, (batch_size,))
    batch = torch.stack([torch.from_numpy((data[i:i + max_length]).astype(np.int64)) for i in ix])
    labels = mask_unrelated(batch)
    return batch, labels


if __name__ == "__main__":
    max_length = 2048
    batch_size = 12
    eval_iters = 100
    device = 'cuda:0'

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/data1/liuyang/llm_pretrain_xmodel/out_xdata_f_line/")
    parser.add_argument("--run_name", type=str, default='data_mix_1_to_1', help="run name")
    parser.add_argument("--tokenizer_version", type=str, default='v6', help="tokenizer version")
    parser.add_argument("--vocab_size", type=int, default=32000, help="vocab size")
    parser.add_argument("--mask_unrelated", action="store_true", help="mask unrelated")
    args = parser.parse_args()

    data = np.memmap(os.path.join('data/in_domain', f'in_domain_{args.tokenizer_version}_{args.vocab_size}.bin'),
                     dtype=np.uint16, mode='r')

    model_dir = os.path.join(args.model_dir, args.run_name)
    tokenizer_path = f'tokenizers/xmodel/{args.tokenizer_version}/xmodel_{args.vocab_size}/'
    model_path = f'working_dir_{args.run_name}'

    dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    dirs = sorted(dirs)

    filename = f'{args.run_name}.jsonl'

    with open(filename, 'w') as fp:
        for ckp_name in tqdm(dirs):
            # print('ckp_name: ' + str(ckp_name))
            ckp_folder = os.path.join(model_dir, ckp_name)
            deploy_model(ckp_folder, args.run_name, tokenizer_path=tokenizer_path)

            model = transformers.AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
            losses = torch.zeros(eval_iters)

            model.to(device)
            model.eval()

            for k in range(eval_iters):
                input_ids, labels = get_batch()
                input_ids = input_ids.to(model.device)
                labels = labels.to(model.device)

                if not args.mask_unrelated:
                    labels = input_ids

                with torch.inference_mode():
                    loss = model(input_ids, labels=labels).loss

                losses[k] = loss.item()

            loss = losses.mean().cpu().item()
            # print('loss: ' + str(loss))

            log_line = json.dumps(dict(ckp_name=ckp_name, loss=loss)) + '\n'
            fp.write(log_line)
            fp.flush()
