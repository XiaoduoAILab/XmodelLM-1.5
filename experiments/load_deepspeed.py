import deepspeed
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from pathlib import Path
import sys

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from megatron import get_args
from megatron.model import GPTModel, GPTModelPipe
from megatron.arguments import core_transformer_config_from_args
from megatron.initialize import initialize_megatron
from megatron.checkpointing import load_checkpoint

initialize_megatron(extra_args_provider=None,
                    args_defaults={})

args = get_args()

config = core_transformer_config_from_args(args)

model = GPTModelPipe(
    config=config,
    num_tokentypes=0,
    parallel_output=True
)

checkpoint_dir = '/data1/liuyang/llm_pretrain_xmodel/out/large_megatron/'

# do the training and checkpoint saving
# model = load_state_dict_from_zero_checkpoint(model, checkpoint_dir) # already on cpu
# submit to model hub or save the model to share with others

iteration = load_checkpoint(checkpoint_dir, optimizer=None, opt_param_scheduler=None)

print(iteration)


