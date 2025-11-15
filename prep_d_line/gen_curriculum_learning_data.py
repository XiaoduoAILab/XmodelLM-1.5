import os
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from data_utils import create_dataloaders

if __name__ == "__main__":
    device = 'cuda:7'
    model_path = '/data1/liuyang/models/c_line/nano'
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.to(device)
    print(model)

    dataset = 'redpajama'
    micro_batch_size = 15
    max_length = 4096
    sampling = 'random'
    data_path = '/data2/liuyang/lit-redpajama-new/'
    total_batches = 100000 * 4 * 30 * 8 // micro_batch_size
    arr_len = total_batches * max_length

    # Data proportions from https://arxiv.org/pdf/2302.13971.pdf Table 1
    data_config = [
        ("arxiv", 2.5),
        ("book", 4.5),
        ("c4", 15.0),
        ("cc", 67.0),
        ("github", 4.5),
        ("stackexchange", 2.0),
        ("wikipedia", 4.5),
    ]

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=max_length,
        data_config=data_config,
        sampling=sampling,
        train_data_dir=data_path,
        val_data_dir=None,
        seed=1337
    )
    dataloader_iter = iter(train_dataloader)
    # batch = next(dataloader_iter).cuda()
    #
    # print('batch: ' + str(batch))
    # print('batch.shape: ' + str(batch.shape))
    #
    # outputs = model(batch, labels=batch)
    # print(outputs.loss)

    data_path = os.path.join(os.path.dirname(__file__), f'curriculum_data.bin')
    ppl_path = os.path.join(os.path.dirname(__file__), f'curriculum_ppl.bin')

    data = np.memmap(data_path, dtype=np.uint16, mode='w+', shape=(arr_len,))
    ppl = np.memmap(ppl_path, dtype=np.float32, mode='w+', shape=(total_batches,))

    for idx in tqdm(range(total_batches), desc=f'writing {data_path}'):
        batch = next(dataloader_iter).to(device)
        chunk_size = max_length * micro_batch_size
        data[idx * chunk_size: (idx + 1) * chunk_size] = torch.flatten(batch).cpu()

        with torch.inference_mode():
            outputs = model(batch, labels=batch)

        ppl[idx] = outputs.loss.cpu().item()

    data.flush()
    ppl.flush()
