# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import json

from datasets import load_dataset  # huggingface datasets
from tqdm import tqdm

if __name__ == "__main__":
    # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
    dataset = load_dataset("openwebtext", data_files=r'C:\datasets\openwebtext.tar.xz')

    with open(r"C:\datasets\openwebtext.json", 'w') as fp:

        for item in tqdm(dataset["train"]):
            line = dict(text=item['text'])
            fp.write(json.dumps(line))
            fp.write('\n')
            fp.flush()
