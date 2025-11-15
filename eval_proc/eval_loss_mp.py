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


def get_batch():
    data = val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    # x = x.pin_memory().to(device, non_blocking=True)

    return x


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model):
    model.eval()

    losses = np.zeros(eval_iters)
    for k in range(eval_iters):
        data = get_batch().to(model.device)
        loss = model(data, labels=data).loss
        losses[k] = loss.item()

    return losses.mean()


def evaluate(item):
    model_name = item['model_name']
    ckp_name = item['ckp_name']
    ckp_folder = os.path.join(checkpoints_dir, ckp_name)
    print(f'worker-{local_rank} start {ckp_folder}')
    deploy_code(ckp_folder)

    model = transformers.AutoModelForCausalLM.from_pretrained(ckp_folder, torch_dtype=torch.bfloat16,
                                                              trust_remote_code=True)
    model.to(f'cuda:{local_rank}')
    model.eval()
    val_loss = estimate_loss(model)
    num_iter = int(ckp_name.split('-')[-1])
    data = dict(model_name=model_name, num_iter=num_iter, val_loss=val_loss)
    json_line = json.dumps(data)
    print(f'worker-{local_rank} done {json_line}')
    return json_line + '\n'


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    import argparse

    parser = argparse.ArgumentParser(description='Theory of Mind')
    # general
    parser.add_argument('--local_rank', type=int, default=0, help='local rank')

    args = parser.parse_args()

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))

    master_process = local_rank == 0


    def print_rank_0(msg):
        if master_process:
            print(msg)


    block_size = 4096
    batch_size = 30
    device = 'cuda'
    eval_iters = 200

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained('tokenizers', use_fast=False, trust_remote_code=True)
    print_rank_0(tokenizer)

    items = []

    for model_name in ['nano', 'micro', 'tiny', 'small']:
        checkpoints_dir = 'out/' + model_name
        dirs = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d))]
        dirs = sorted(dirs)
        # print(dirs)
        for ckp_name in dirs:
            items.append(dict(model_name=model_name, ckp_name=ckp_name))

    items.sort(key=lambda x: x['model_name'] + x['ckp_name'])
    lines = []
    num_items = len(items)

    if master_process:
        t = tqdm(range(local_rank, num_items, world_size))
    else:
        t = range(local_rank, num_items, world_size)

    for i in t:
        item = items[i]
        lines.append(evaluate(item))

    with open(f'estimate_loss_{local_rank}.jsonl', 'w') as fp:
        fp.writelines(lines)
