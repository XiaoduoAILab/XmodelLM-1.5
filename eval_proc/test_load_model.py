import sys
from pathlib import Path
import torch
import sys
from pathlib import Path

import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from models.modeling_xmodel import XModelForCausalLM
from models.configuration_xmodel import XModelConfig

# model = torch.load()
# print(model)
model_name = 'medium'
config = XModelConfig.from_name(model_name)
print(config)
model = XModelForCausalLM(config)
PATH = 'working_dir/pytorch_model.bin'
model.load_state_dict(torch.load(PATH))

#
#
#
#
#
#
# # load tokenizer
# tokenizer = transformers.AutoTokenizer.from_pretrained('working_dir', use_fast=False, trust_remote_code=True)
# print(tokenizer)
#
# model = XModelForCausalLM.from_pretrained('working_dir',
#                                                           trust_remote_code=True)
#
# print(model)
