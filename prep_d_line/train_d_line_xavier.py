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
import math
import os
import sys
import time
from contextlib import contextmanager
from contextlib import nullcontext
from math import exp
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from models.modeling_xmodel import XModelForCausalLM
from models.configuration_xmodel import XModelConfig
from utils import setup_seeds, calculate_norm, calculate_sparsity, calculate_orthog_mse, \
    calculate_l2_norm, calculate_grad_norm, print_model_ratio, choose_last_iter
from data_utils import create_dataloaders


def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=-1, help="Reserved for deepspeed framework")
    parser.add_argument("--model_name", type=str, default='medium', help="model name")
    parser.add_argument("--run_name", type=str, default=None, help="run name")
    parser.add_argument("--data_path", type=str, default="/data2/liuyang/lit-redpajama-new/", help="data set path")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--dtype", type=str, default='bfloat16', help="data type")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--min_lr", type=float, default=None, help="minimal learning rate")
    parser.add_argument("--optimizer_eps", type=float, default=1e-5, help="optimizer eps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
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
    parser.add_argument("--calculate_ortho", action="store_true", help="calculate ortho")
    parser.add_argument("--initializer_range", type=float, default=0.02, help="initializer range")
    parser.add_argument("--cache_dir", type=str, default='/data1/liuyang/.cache', help="cache dir")
    parser.add_argument("--device_model", type=str, default=None, help="one of [A100, H100, DCU, V100]")
    parser.add_argument("--compile", action="store_true", help="compile")
    parser.add_argument("--sampling", type=str, default='random', help="sampling method")
    parser.add_argument("--resume", action="store_true", help="resume")
    parser.add_argument("--stop_steps", type=int, default=None, help="stop steps")
    return parser


def get_batch(split):
    if split == 'train':
        return next(dataloader_iter).to(device)
    elif split == 'OpenWebText2':
        data = val_data
    elif split == 'LAMBADA':
        data = lambada
    elif split == 'PTB':
        data = ptb
    elif split == 'WikiText2':
        data = wikitext2

    ix = torch.randint(len(data) - args.max_length, (batch_size,))
    batch = torch.stack([torch.from_numpy((data[i:i + args.max_length]).astype(np.int64)) for i in ix])
    # pin arrays batch, which allows us to move them to GPU asynchronously (non_blocking=True)
    batch = batch.pin_memory().to(device, non_blocking=True)
    return batch


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'OpenWebText2', 'LAMBADA', 'PTB', 'WikiText2']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            batch = get_batch(split)
            with ctx:
                loss = model(batch, labels=batch).loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


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


def train(iter_num):
    global batch, raw_model, dt_get_batch, dt
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    num_tokens_trained = iter_num * tokens_per_iter

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        with catch_time() as elapsed_time:
            losses = estimate_loss()

        writer.add_scalar("Time/estimate_loss", elapsed_time(), num_tokens_trained)
        # print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        writer.add_scalar("Main/train_loss", losses['train'], num_tokens_trained)
        writer.add_scalar("Main/val_loss", losses['OpenWebText2'], num_tokens_trained)

        for key in losses.keys():
            if key != 'train':
                writer.add_scalar(f"PPL/{key}", exp(losses[key]), num_tokens_trained)

        with catch_time() as elapsed_time:
            mae, mse = calculate_norm(model)

        writer.add_scalar("Time/calculate_norm", elapsed_time(), num_tokens_trained)
        writer.add_scalar("Norm/MAE", mae, num_tokens_trained)
        writer.add_scalar("Norm/MSE", mse, num_tokens_trained)

        with catch_time() as elapsed_time:
            l2_norm_dict = calculate_l2_norm(model)

        writer.add_scalar("Time/calculate_l2_norm", elapsed_time(), num_tokens_trained)
        for key in l2_norm_dict.keys():
            writer.add_scalar(f"Norm/l2 norm ({key})", l2_norm_dict[key], num_tokens_trained)

        with catch_time() as elapsed_time:
            sparsity_dict = calculate_sparsity(model)

        writer.add_scalar("Time/calculate_sparsity", elapsed_time(), num_tokens_trained)

        for key in sparsity_dict.keys():
            writer.add_scalar('Sparsity/' + key, sparsity_dict[key], num_tokens_trained)

        if args.calculate_ortho:
            with catch_time() as elapsed_time:
                ortho_mse_dict = calculate_orthog_mse(model)

            writer.add_scalar("Time/calculate_ortho_mse", elapsed_time(), num_tokens_trained)

            for key in ortho_mse_dict.keys():
                writer.add_scalar('Orthogonality/' + key, ortho_mse_dict[key], num_tokens_trained)

        if always_save_checkpoint:

            if iter_num > 0:
                save_folder = os.path.join(out_dir, model_name)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                save_folder = os.path.join(save_folder, f"iter-{iter_num:06d}")
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                print(f"saving checkpoint to {save_folder}")

                raw_model.save_pretrained(save_folder, is_main_process=master_process, safe_serialization=False)

                checkpoint = {
                    # 'optimizer': optimizer.state_dict(),
                    'model_args': vars(args),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                torch.save(checkpoint, os.path.join(save_folder, 'ckpt.pt'))

    t0 = time.time()
    dt_get_batch = 0
    losses = []

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            loss = model(batch, labels=batch).loss
            loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation

        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        losses.append(loss)

        t0_get_batch = time.time()
        batch = get_batch('train')
        dt_get_batch += time.time() - t0_get_batch

    if iter_num % eval_interval == 0 and master_process:
        # Compute the global L2 gradient norm
        with catch_time() as elapsed_time:
            grad_norm = calculate_grad_norm(model)

        writer.add_scalar("Time/calculate_grad_norm", elapsed_time(), num_tokens_trained)
        writer.add_scalar("Norm/grad norm", grad_norm, num_tokens_trained)

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()

    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0

    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        # lossf = loss.item() * gradient_accumulation_steps
        lossf = sum(losses)

        bar_desc = f"loss: {lossf:.4f}"
        t.set_description(bar_desc, refresh=True)

        flops_achieved, mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt,
                                                     max_length=max_length, device_model=args.device_model)

        writer.add_scalar("Main/flops_achieved", flops_achieved * ddp_world_size, num_tokens_trained)
        writer.add_scalar("Main/mfu", mfu * 100, num_tokens_trained)
        writer.add_scalar("Main/throughput", tokens_per_iter / dt, num_tokens_trained)
        writer.add_scalar("Main/loss", lossf, num_tokens_trained)
        writer.add_scalar("Main/loss scale", scaler.get_scale(), num_tokens_trained)
        writer.add_scalar("Main/lr", lr, num_tokens_trained)
        writer.add_scalar("Time/get batch", dt_get_batch, num_tokens_trained)


if __name__ == "__main__":
    setup_seeds(42)
    # torch.autograd.set_detect_anomaly(True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers
    dataset = 'xdata_d_line'
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    model_name = args.model_name
    eval_interval = args.eval_interval
    gradient_accumulation_steps = args.gradient_accumulation_steps
    batch_size = args.micro_batch_size
    max_length = args.max_length
    learning_rate = args.learning_rate
    optimizer_eps = args.optimizer_eps
    if args.min_lr is None:
        min_lr = learning_rate / 10  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    else:
        min_lr = args.min_lr
    total_steps = args.total_steps
    lr_decay_iters = total_steps  # should be ~= max_iters per Chinchilla
    warmup_iters = args.warmup_steps
    out_dir = 'out_' + dataset
    log_interval = 1
    # eval_iters = 200
    eval_iters = 100
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
    device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    # dtype = 'bfloat16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    dtype = args.dtype
    # compile = True  # use PyTorch 2.0 to compile the model to be faster
    compile = args.compile

    # ==============================
    # Initialize Distributed Training
    # ==============================
    ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?

    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # assert gradient_accumulation_steps % torch.cuda.device_count() == 0
        # gradient_accumulation_steps //= torch.cuda.device_count()
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1


    def print_rank_0(msg):
        if master_process:
            print(msg)


    print_rank_0(args)

    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_length
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
    val_data = np.memmap(os.path.join('data/openwebtext', 'val.bin'), dtype=np.uint16, mode='r')
    lambada = np.memmap('data/lambada/lambada.bin', dtype=np.uint16, mode='r')
    ptb = np.memmap('data/ptb/ptb.bin', dtype=np.uint16, mode='r')
    wikitext2 = np.memmap('data/wikitext2/wikitext2.bin', dtype=np.uint16, mode='r')

    # Data proportions from https://arxiv.org/pdf/2302.13971.pdf Table 1
    data_config = [
        ("arxiv", 2.5),
        ("book", 4.5),
        ("c4", 15.0),
        ("v2", 67.0),
        ("github", 4.5),
        ("stackexchange", 2.0),
        ("wiki", 4.5),
    ]

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=args.micro_batch_size,
        block_size=args.max_length,
        data_config=data_config,
        sampling=args.sampling,
        train_data_dir=args.data_path,
        val_data_dir=None,
        seed=1337 + seed_offset
    )

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    start_iter = 0
    best_val_loss = 1e9

    # logging
    log_dir = 'runs/' + model_name + '_' + dataset
    if args.run_name is not None:
        log_dir += '_' + args.run_name
    writer = SummaryWriter(log_dir)

    # ==============================
    # Initialize Model, Optimizer and LR Scheduler
    # ==============================
    config = XModelConfig.from_name(model_name)
    print_rank_0(config)


    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


    model = XModelForCausalLM(config)
    model.apply(init_weights)

    print_rank_0(model)

    if args.resume:
        save_folder = os.path.join(out_dir, model_name)
        last_iter = choose_last_iter(save_folder)
        save_folder = os.path.join(save_folder, last_iter)

        print_rank_0(f"Resuming training from {save_folder}")
        # resume training from a checkpoint.

        ckpt_path = os.path.join(save_folder, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

        PATH = os.path.join(save_folder, 'pytorch_model.bin')
        model.load_state_dict(torch.load(PATH, map_location=device))

        start_iter = iter_num + 1
    else:
        # init a new model from scratch
        print_rank_0("Initializing a new model from scratch")

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        print_rank_0("Gradient-checkpointing enabled successfully")

    if args.use_flash_attn:
        from patch.flash_attention_patch import replace_with_flash_attention

        replace_with_flash_attention(model=model)
        print_rank_0("Flash-attention enabled successfully")

    total_params = sum(p.numel() for p in model.parameters())
    print_rank_0('Total params: ' + str(total_params))
    flops_per_token = model.flops_per_token(max_length=max_length)
    print_rank_0('flops per token: ' + str(flops_per_token))

    if master_process:
        print_model_ratio(model)

    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        foreach=False,
        eps=optimizer_eps
        # fused=False
    )

    # compile the model
    if compile:
        print_rank_0("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # training loop
    dataloader_iter = iter(train_dataloader)
    batch = get_batch('train')  # fetch the very first batch

    raw_model = model.module if ddp else model  # unwrap DDP container if needed
    dt_get_batch = float('inf')
    dt = float('inf')

    scope = range(start_iter, total_steps + 1)
    if master_process:
        t = tqdm(scope, initial=start_iter, total=total_steps + 1, desc='', leave=True)
    else:
        t = scope
    for iter_num in t:
        train(iter_num)

        if args.stop_steps is not None and iter_num >= args.stop_steps:
            break

    if ddp:
        destroy_process_group()
