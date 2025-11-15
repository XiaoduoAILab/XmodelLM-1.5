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
import functools
import math
import os
import sys
from contextlib import contextmanager
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from models.modeling_xmodel import XModelForCausalLM, Model
from models.configuration_xmodel import XModelConfig
from utils import setup_seeds, print_model_ratio


def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=-1, help="Reserved for deepspeed framework")

    parser.add_argument("--model_name", type=str, default='medium', help="model name")
    parser.add_argument("--run_name", type=str, default=None, help="run name")
    parser.add_argument("--dtype", type=str, default='bfloat16', help="data type")

    parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--optimizer_eps", type=float, default=1e-5, help="optimizer eps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--micro_batch_size", type=int, default=10, help="train micro batch size per gpu.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="weight decay")
    parser.add_argument("--total_steps", type=int, default=10000, help="total steps")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="warmup steps")
    parser.add_argument("--grad_checkpoint", action="store_true", help="use gradient checkpointing")
    parser.add_argument("--max_length", type=int, default=4096, help="block size")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clipping")
    parser.add_argument("--eval_interval", type=int, default=100, help="eval interval")
    parser.add_argument("--use_flash_attn", type=bool, default=False, help="use flash attn")
    parser.add_argument("--noise_scale", action="store_true", help="use noise scale")
    parser.add_argument("--calculate_ortho", action="store_true", help="calculate ortho")
    parser.add_argument("--initializer_range", type=float, default=0.02, help="initializer range")
    parser.add_argument("--device_model", type=str, default='DCU', help="one of [A100, H100, DCU, V100]")
    parser.add_argument("--compile", action="store_true", help="compile")
    parser.add_argument("--job_id", type=str, default=0, help="slurm job id")
    parser.add_argument("--node_size", type=int, default=1, help="parallel node number")
    parser.add_argument("--line_index", type=str, default='line/', help="tensorbord log data in which line")
    return parser


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - max_length, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + max_length]).astype(np.int64)) for i in ix])
    # pin arrays x, which allows us to move them to GPU asynchronously (non_blocking=True)
    x = x.pin_memory().to(device, non_blocking=True)
    return x


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


@contextmanager
def catch_time() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


def train(iter_num, model, optimizer):
    global batch
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Compute and accumulate the gradients
    model.zero_grad()
    losses = []

    for micro_step in range(gradient_accumulation_steps):
        model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

        with ctx:
            loss = model(batch, labels=batch).loss
            loss = loss / gradient_accumulation_steps

        loss.backward()
        losses.append(loss)

        batch = get_batch('train').to(device)

    acc_loss = sum(losses)

    optimizer.step()

    return acc_loss


def setup(rank, world_size):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def fsdp_main(rank, world_size, args):
    setup(rank, world_size)

    # my_auto_wrap_policy = functools.partial(
    #     size_based_auto_wrap_policy, min_num_params=1000000
    # )
    print('step_flag_0')
    torch.cuda.set_device(rank)
    xmodel_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Model,
        }
    )

    sharding_strategy: ShardingStrategy = ShardingStrategy.SHARD_GRAD_OP  # for Zero2 and FULL_SHARD for Zero3

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    # ==============================
    # Initialize Model, Optimizer and LR Scheduler
    # ==============================
    config = XModelConfig.from_name(model_name)
    print_rank_0(config)
    print('step_flag_1')
    model = XModelForCausalLM(config)
    print_rank_0(model)
    print('step_flag_2')
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        print_rank_0("Gradient-checkpointing enabled successfully")

    if args.use_flash_attn:
        from patch.flash_attention_patch import replace_with_flash_attention

        replace_with_flash_attention(model=model)
        print_rank_0("Flash-attention enabled successfully")

    total_params = sum(p.numel() for p in model.parameters())
    print_rank_0('Total params: ' + str(total_params))
    # effective_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print_rank_0('Effective params: ' + str(effective_params))
    flops_per_token = model.flops_per_token(max_length=max_length)
    print_rank_0('flops per token: ' + str(flops_per_token))

    if master_process:
        print_model_ratio(model)

    model.to(device)
    # compile the model
    if compile:
        print_rank_0("compiling the model... (takes a ~minute)")
        model = torch.compile(model)  # requires PyTorch 2.0

    bf16 = MixedPrecision(
        param_dtype=torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision.
        buffer_dtype=torch.bfloat16,
    )

    # wrap model into DDP container
    fsdp_model = FSDP(model,
                      auto_wrap_policy=xmodel_auto_wrap_policy,
                      # mixed_precision=bf16,
                      sharding_strategy=sharding_strategy,
                      device_id=torch.cuda.current_device())

    # optimizer
    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        foreach=False,
        eps=optimizer_eps
        # fused=False
    )

    init_start_event.record()
    # training loop
    if master_process:
        t = trange(total_steps, desc='', leave=True)
    else:
        t = range(total_steps)
    for iter_num in t:
        num_tokens_trained = iter_num * tokens_per_iter

        loss = train(iter_num, fsdp_model, optimizer)

        if master_process:
            # lossf = loss.item() * gradient_accumulation_steps
            lossf = loss.item()
            bar_desc = f"loss: {lossf:.4f}"
            t.set_description(bar_desc, refresh=True)
            writer.add_scalar("Main/loss", lossf, num_tokens_trained)

    init_end_event.record()
    cleanup()


if __name__ == "__main__":
    setup_seeds(42)
    # torch.autograd.set_detect_anomaly(True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers
    dataset = 'openwebtext'
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

    # I/O
    out_dir = 'out'
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
    # DDP settings
    backend = 'nccl'  # 'nccl', 'gloo', etc.
    # system
    # device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    # dtype = 'bfloat16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    dtype = args.dtype
    # compile = True  # use PyTorch 2.0 to compile the model to be faster
    compile = args.compile

    # ==============================
    # Initialize Distributed Training
    # ==============================
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    master_process = rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = rank  # each process gets a different seed


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
    device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # data

    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    best_val_loss = 1e9

    # init a new model from scratch
    print_rank_0("Initializing a new model from scratch")

    # logging
    # log_dir = 'runs/' + model_name + '_' + dataset
    log_dir = 'runs/' + line_index + model_name + '_' + dataset + '_' + job_id + '_' + str(node_size)
    if args.run_name is not None:
        log_dir += '_' + args.run_name
    writer = SummaryWriter(log_dir)
    print('step_flag_000')
    batch = get_batch('train').to(device)  # fetch the very first batch
    WORLD_SIZE = torch.cuda.device_count()
    print('step_flag_000111')
    fsdp_main(rank, world_size, args)
