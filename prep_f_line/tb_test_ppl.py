import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from tqdm import tqdm

wd = Path(__file__).parent.parent.resolve()
print('wd: ' + str(wd))
sys.path.append(str(wd))

from eval_proc.utils import deploy_model


def get_batch():
    ix = torch.randint(len(data) - max_length, (batch_size,))
    batch = torch.stack([torch.from_numpy((data[i:i + max_length]).astype(np.int64)) for i in ix])
    return batch, batch


if __name__ == "__main__":
    max_length = 4096
    batch_size = 12
    eval_iters = 100
    device = 'cuda:0'
    model_name = 'xl'

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/data2/liuyang/llm_pretrain_xmodel/out_f_line/xl_f_line")
    parser.add_argument("--language", type=str, default='ar', help="language")
    parser.add_argument("--tokenizer_version", type=str, default='v11', help="tokenizer version")
    parser.add_argument("--vocab_size", type=int, default=65280, help="vocab size")
    args = parser.parse_args()

    data = np.memmap(
        os.path.join('data/multilingual', f'{args.language}_{args.tokenizer_version}_{args.vocab_size}.bin'),
        dtype=np.uint16, mode='r')

    model_dir = args.model_dir
    tokenizer_path = f'tokenizers/xmodel/{args.tokenizer_version}/xmodel_{args.vocab_size}/'
    model_path = f'working_dir_xl'

    dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    dirs = sorted(dirs)

    filename = f'{args.language}.jsonl'

    with open(filename, 'w') as fp:
        for ckp_name in tqdm(dirs):
            # print('ckp_name: ' + str(ckp_name))
            ckp_folder = os.path.join(model_dir, ckp_name)
            deploy_model(ckp_folder, model_name, tokenizer_path=tokenizer_path)

            model = transformers.AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
            model.to(device)
            model.eval()

            losses = torch.zeros(eval_iters)

            for k in range(eval_iters):
                input_ids, labels = get_batch()
                input_ids = input_ids.to(model.device)
                labels = labels.to(model.device)

                with torch.inference_mode():
                    loss = model(input_ids, labels=labels).loss

                losses[k] = loss.item()

            loss = losses.mean().cpu().item()
            # print('loss: ' + str(loss))

            log_line = json.dumps(dict(ckp_name=ckp_name, loss=loss)) + '\n'
            fp.write(log_line)
            fp.flush()
