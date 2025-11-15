import sys
from pathlib import Path

import torch
import transformers

from models.configuration_xmodel import XModelConfig
from models.modeling_xmodel import XModelForCausalLM

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from utils import extract_embeddings

if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained('working_dir', use_fast=False, trust_remote_code=True)
    print(tokenizer)

    model_name = 'small'
    config = XModelConfig.from_name(model_name)
    print(config)
    model = XModelForCausalLM(config)
    PATH = 'working_dir/pytorch_model.bin'
    model.load_state_dict(torch.load(PATH))

    texts = [
        "this is a test",
        "this is another test case with a different length",
    ]
    print(texts)

    embeddings = extract_embeddings(model, tokenizer, texts)
    print(embeddings.shape)
    print(embeddings)
