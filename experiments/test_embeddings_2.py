import sys
from pathlib import Path

import torch
import transformers
from sentence_transformers import SentenceTransformer

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained('working_dir', use_fast=False, trust_remote_code=True)
    print(tokenizer)

    model = SentenceTransformer('working_dir')
    PATH = 'working_dir/pytorch_model.bin'
    model.load_state_dict(torch.load(PATH))

    sentences = [
        "this is a test",
        "this is another test case with a different length",
    ]
    print(sentences)

    embeddings = model.encode(sentences)
    print(embeddings.shape)
    print(embeddings)
