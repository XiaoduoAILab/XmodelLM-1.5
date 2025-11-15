import json
import os

import numpy as np
import torch
import transformers
from tqdm import tqdm

from utils import deploy_code

# load data
dataset = 'openwebtext'
data_dir = os.path.join('data', dataset)
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')


def get_batch(split):
    data = val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    x = x.pin_memory().to(device, non_blocking=True)

    return x


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['val']:
        losses = np.zeros(eval_iters)
        for k in tqdm(range(eval_iters)):
            data = get_batch(split)
            loss = model(data, labels=data).loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out


def evaluate(item):
    model_name = item['model_name']
    ckp_folder = item['ckp_folder']
    ckp_name = item['ckp_name']
    deploy_code(ckp_folder)

    model = transformers.AutoModelForCausalLM.from_pretrained(ckp_folder, torch_dtype=torch.float16,
                                                              trust_remote_code=True)
    model.cuda()
    model.eval()
    losses = estimate_loss(model)
    num_iter = int(ckp_name.split('-')[-1])
    item = dict(model_name=model_name, num_iter=num_iter, val_loss=losses['val'])

    json_line = json.dumps(item)
    return json_line + '\n'


if __name__ == "__main__":
    block_size = 4096
    batch_size = 30
    device = 'cuda'
    device_type = 'cuda'
    eval_iters = 200
    ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--checkpoints_dir", type=str, default="../experiments/llm_pretrain_xmodel/out/")
    #
    # args = parser.parse_args()

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained('tokenizers', use_fast=False, trust_remote_code=True)
    print(tokenizer)

    items = []

    for model_name in ['nano', 'micro', 'tiny', 'small']:
        checkpoints_dir = 'out/' + model_name
        dirs = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d))]
        dirs = sorted(dirs)
        # print(dirs)
        for ckp_name in dirs:
            ckp_folder = os.path.join(checkpoints_dir, ckp_name)
            # print(f'estimate_loss <- {ckp_folder}')
            items.append(dict(model_name=model_name, ckp_folder=ckp_folder, ckp_name=ckp_name))

    lines = []
    for item in tqdm(items):
        lines.append(evaluate(item))

    with open(f'estimate_loss.jsonl', 'w') as fp:
        fp.writelines(lines)
