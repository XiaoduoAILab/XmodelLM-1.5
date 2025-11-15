# Copyright 2023 XiaoDuo AI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import sys
import time
from pathlib import Path
import deepspeed
import torch
from torch.utils.data.distributed import DistributedSampler

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from models.modeling_xmodel import XModelForCausalLM
from models.configuration_xmodel import XModelConfig
from utils import setup_seeds

# from Pytorch_Memory_Utils.gpu_mem_track import MemTracker
from models.modeling_xmodel_pipe import get_layers_from_config, loss_fn, PipeDataset
from deepspeed.pipe import PipelineModule
from torch.utils.data import DataLoader


def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=-1, help="Reserved for deepspeed framework")
    parser.add_argument("--model_name", type=str, default='medium', help="model name")
    parser.add_argument("--run_name", type=str, default=None, help="run name")
    parser.add_argument("--data_path", type=str,
                        default=str(wd)+"/data/redpajama-data-1T/red_pajama/",
                        help="data set path")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--dtype", type=str, default='bfloat16', help="data type")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--optimizer_eps", type=float, default=1e-5, help="optimizer eps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--micro_batch_size", type=int, default=10, help="train micro batch size per gpu.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="weight decay")
    parser.add_argument("--total_steps", type=int, default=350000, help="total steps")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="warmup steps")
    parser.add_argument("--grad_checkpoint", action="store_true", help="use gradient checkpointing")
    parser.add_argument("--max_length", type=int, default=4096, help="max sequence length")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clipping")
    parser.add_argument("--eval_interval", type=int, default=100, help="eval interval")
    parser.add_argument("--use_flash_attn", type=bool, default=False, help="use flash attn")
    parser.add_argument("--noise_scale", action="store_true", help="use noise scale")
    parser.add_argument("--calculate_ortho", action="store_true", help="calculate ortho")
    parser.add_argument("--initializer_range", type=float, default=0.02, help="initializer range")
    parser.add_argument("--cache_dir", type=str, default='/data1/liuyang/.cache', help="cache dir")
    parser.add_argument("--device_model", type=str, default='DCU', help="one of [A100, H100, DCU, V100]")
    parser.add_argument("--compile", action="store_true", help="compile")
    parser.add_argument("--sampling", type=str, default='in_order', help="sampling method")
    parser.add_argument("--resume", action="store_true", help="compile")
    parser.add_argument("--job_id", type=str, default=0, help="slurm job id")
    parser.add_argument("--node_size", type=int, default=1, help="parallel node number")
    parser.add_argument("--line_index", type=str, default='line/', help="tensorbord log data in which line")
    parser.add_argument("--num_stages",type=int, default=2, help="pipeline parallel stages")

    return parser

def train_pipe():
    config = XModelConfig.from_name(model_name)
    print_rank_0(config)
    # gpu_tracker.track()
    model = XModelForCausalLM(config)
    pipe_model = get_layers_from_config( config)  # pipe_model
    stack_model = PipelineModule(layers=pipe_model, num_stages=num_stages,
                                 partition_method='parameters',
                                 activation_checkpoint_interval=0,
                                 loss_fn=loss_fn
                                 )
    print('stack_model_stack_model111')
    print_rank_0(stack_model)
    # pdb.set_trace()
    total_params = sum(p.numel() for p in model.parameters())
    print_rank_0('Total params: ' + str(total_params))
    flops_per_token = model.flops_per_token(max_length=max_length)
    print_rank_0('flops per token: ' + str(flops_per_token))

    optimizer = torch.optim.AdamW(
        params=stack_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        eps=optimizer_eps
        # fused=False
    )

    data_config = [
        # ("arxiv", 2.5),
        # ("book", 4.5),
        # ("c4", 15.0),
        # ("cc", 67.0),
        # ("github", 4.5),
        # ("stackexchange", 2.0),
        ("wikipedia", 4.5),
    ]
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    pipe_dataset = PipeDataset(data_config, args.data_path, max_length)
    train_sampler = DistributedSampler(pipe_dataset, shuffle=True)
    if args.local_rank in [-1, 0]:
        torch.distributed.barrier()
    # train_sampler = DistributedSampler(pipe_dataset, shuffle=True)
    pipe_dataloader = DataLoader(dataset=pipe_dataset, batch_size=batch_size,sampler=train_sampler)
    pipe_dataiter = iter(pipe_dataloader)

    # data_config = [
    #     ("arxiv", 2.5),
    #     # ("book", 4.5),
    #     # ("c4", 15.0),
    #     # ("cc", 67.0),
    #     # ("github", 4.5),
    #     # ("stackexchange", 2.0),
    #     ("wikipedia", 4.5),
    # ]
    # pipe_dataloader, val_dataloader = create_dataloaders(
    #     batch_size=args.micro_batch_size,
    #     block_size=args.max_length,
    #     data_config=data_config,
    #     sampling=args.sampling,
    #     train_data_dir=args.data_path,
    #     val_data_dir=None,
    #     seed=1337 + seed_offset
    # )
    dataloader_iter = iter(pipe_dataloader)

    # stack_model_parameters = filter(lambda p: p.requires_grad, stack_model.parameters())
    print('stack_model_stack_model222',stack_model)
    # for name, param in stack_model.named_parameters():
        # if param.requires_grad:
        #     print('require_grad_para_name',name)
        # else:
        #     print('non_grad_para_name', name)
    model_engine, _, _, _ = deepspeed.initialize(args=args,
                                                 model=stack_model,
                                                 model_parameters=stack_model.parameters(),
                                                 optimizer=optimizer,
                                                 config=ds_config,
                                                 # training_data=pipe_dataset
                                                 )
    # pdb.set_trace()
    for i in range(10):
        loss = model_engine.train_batch(data_iter=dataloader_iter)
        # loss = model_engine.train_batch()
        print('loss',loss.item())


if __name__ == "__main__":
    setup_seeds(42)
    # torch.autograd.set_detect_anomaly(True)
    # gpu_tracker = MemTracker()
    # gpu_tracker.track()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers
    dataset = 'redpajama'
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    model_name = args.model_name
    eval_interval = args.eval_interval
    gradient_accumulation_steps = args.gradient_accumulation_steps
    batch_size = args.micro_batch_size
    max_length = args.max_length
    learning_rate = args.learning_rate
    optimizer_eps = args.optimizer_eps
    min_lr = learning_rate / 10  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    total_steps = args.total_steps
    lr_decay_iters = total_steps  # should be ~= max_iters per Chinchilla
    warmup_iters = args.warmup_steps
    job_id = args.job_id
    node_size = args.node_size
    line_index = args.line_index
    num_stages=args.num_stages
    out_dir = 'out_' + dataset
    log_interval = 1
    eval_iters = 200
    eval_only = False  # if True, script exits right after the first eval
    always_save_checkpoint = True  # if True, always save a checkpoint after each eval

    # adamw optimizer
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True  # whether to decay the learning rate
    # system
    device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    # dtype = 'bfloat16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    dtype = args.dtype
    # compile = True  # use PyTorch 2.0 to compile the model to be faster
    compile = args.compile

    # ==============================
    # Initialize Distributed Training
    # ==============================
    deepspeed.init_distributed()
    ddp_world_size = int(os.environ['WORLD_SIZE'])

    # master_process = args.local_rank == 0
    # seed_offset = args.local_rank

    # print('OMPI_COMM_WORLD_LOCAL_RANK', os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK'), os.environ.get('LOCAL_RANK'))
    # os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = os.environ.get('LOCAL_RANK')
    world_rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
    ddp_rank = world_rank % torch.cuda.device_count()
    device_flag = f'cuda:{ddp_rank}'
    torch.cuda.set_device(device_flag)
    args.local_rank = ddp_rank
    master_process = world_rank == 0
    seed_offset = world_rank
    # print('args_local_rank_master_process111', args.local_rank, master_process)
    print('args_local_rank_master_process222', args.local_rank, master_process)
    print('world_rank_world_size_ddp_rank', world_rank, world_size, ddp_rank)

    ds_config = json.load(open(f'config/deepspeed_{model_name}1.json'))
    print('ds_config',ds_config)

    def print_rank_0(msg):
        if master_process:
            print(msg)


    print_rank_0(args)

    tokens_per_iter = gradient_accumulation_steps * world_size * batch_size * max_length
    print_rank_0(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    # data

    # Data proportions from https://arxiv.org/pdf/2302.13971.pdf Table 1

    iter_num = 0
    best_val_loss = 1e9

    # init a new model from scratch
    print_rank_0("Initializing a new model from scratch")

    # logging
    # ds_logger.setLevel(logging.DEBUG)
    log_dir = 'runs/' + line_index + model_name + '_' + dataset + '_' + job_id + '_' + str(node_size)
    if args.run_name is not None:
        log_dir += '_' + args.run_name

    deepspeed.init_distributed('nccl')
    args.local_rank = int(os.environ['LOCAL_RANK'])
    print('OMPI_COMM_WORLD_LOCAL_RANK', os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK'), os.environ.get('LOCAL_RANK'))
    os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = os.environ.get('LOCAL_RANK')
    ddp_rank=int(os.environ['LOCAL_RANK'])
    device_flag = f'cuda:{ddp_rank}'
    torch.cuda.set_device(device_flag)
    t0 = time.time()

    # if args.pipeline_parallel_size == 0:
    #     # train_base(args)
    #     pass
    # else:
    train_pipe()

    t1 = time.time()
    ttt = t1 - t0
    print('ttt', ttt)