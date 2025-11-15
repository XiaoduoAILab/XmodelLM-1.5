# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from datasets import concatenate_datasets, Dataset

import numpy as np
import transformers
from datasets import load_dataset  # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

# we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)

tokenizer = transformers.AutoTokenizer.from_pretrained('tokenizers/xmodel/v6/xmodel_32000',
                                                       skip_special_tokens=True,
                                                       add_bos_token=False,
                                                       add_eos_token=False,
                                                       clean_up_tokenization_spaces=True,
                                                       use_fast=False,
                                                       trust_remote_code=True)
# print(tokenizer)

if __name__ == "__main__":
    arrow_files = ["data/wikitext2/wikitext-test.arrow"]

    dataset = concatenate_datasets([Dataset.from_file(arrow_file) for arrow_file in arrow_files])["text"]

    encodings = tokenizer("\n\n".join(dataset), return_tensors="pt")
    arr_len = encodings.input_ids.size(1)

    filename = os.path.join(os.path.dirname(__file__), f'wikitext2_v6_32000.bin')
    dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 1024

    arr[:] = encodings.input_ids[:]
    arr.flush()
