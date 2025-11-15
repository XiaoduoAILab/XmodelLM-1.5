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
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

# from patch.flash_attention_patch import replace_with_flash_attention
from models.modeling_xmodel import XModelForCausalLM
from models.configuration_xmodel import XModelConfig
from utils import setup_seeds, calculate_norm, calculate_sparsity, calculate_orthog_mse, \
    calculate_l2_norm, calculate_grad_norm, print_model_ratio
from bs_finder import BSFinder


def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=-1, help="Reserved for deepspeed framework")

    parser.add_argument("--model_name", type=str, default='nano', help="model name")
    parser.add_argument("--run_name", type=str, default=None, help="run name")
    parser.add_argument("--dtype", type=str, default='bfloat16', help="data type")

    parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--optimizer_eps", type=float, default=1e-5, help="optimizer eps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--micro_batch_size", type=int, default=2, help="train micro batch size per gpu.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="weight decay")
    parser.add_argument("--total_steps", type=int, default=10000, help="total steps")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="warmup steps")
    parser.add_argument("--grad_checkpoint", action="store_true", help="use gradient checkpointing")
    parser.add_argument("--block_size", type=int, default=4096, help="block size")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clipping")
    parser.add_argument("--eval_interval", type=int, default=100, help="eval interval")
    parser.add_argument("--use_flash_attn", type=bool, default=False, help="use flash attn")
    parser.add_argument("--noise_scale", action="store_true", help="use noise scale")
    parser.add_argument("--calculate_ortho", action="store_true", help="calculate ortho")
    parser.add_argument("--initializer_range", type=float, default=0.02, help="initializer range")
    parser.add_argument("--job_id", type=str, default=0, help="slurm job id")
    parser.add_argument("--node_size", type=int, default=1, help="parallel node number")
    parser.add_argument("--line_index", type=str, default='line/', help="tensorbord log data in which line")

    return parser


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    # pin arrays x, which allows us to move them to GPU asynchronously (non_blocking=True)
    x = x.pin_memory().to(device, non_blocking=True)
    return x


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            data = get_batch(split)
            with ctx:
                loss = model(data, labels=data).loss
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
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


if __name__ == "__main__":
    setup_seeds(42)
    # torch.autograd.set_detect_anomaly(True)

    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    model_name = args.model_name
    eval_interval = args.eval_interval
    gradient_accumulation_steps = args.gradient_accumulation_steps
    batch_size = args.micro_batch_size
    block_size = args.block_size
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
    device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    # dtype = 'bfloat16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    dtype = args.dtype
    compile = False  # use PyTorch 2.0 to compile the model to be faster

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

    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
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
    dataset = 'openwebtext'
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    # train_data = np.memmap(os.path.join(data_dir, 'owt_8gram.bin'), dtype=np.uint16, mode='r')
    # train_data = np.memmap(os.path.join(data_dir, 'owt_8gram_tok.bin'), dtype=np.uint16, mode='r')
    # train_data = np.memmap(os.path.join(data_dir, 'owt_12gram_sent.bin'), dtype=np.uint16, mode='r')
    # train_data = np.memmap(os.path.join(data_dir, 'owt_16gram_tok.bin'), dtype=np.uint16, mode='r')
    # train_data = np.memmap(os.path.join(data_dir, 'owt_16gram_sent.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # init a new model from scratch
    print_rank_0("Initializing a new model from scratch")

    # logging
    # time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    log_dir = 'runs/' + line_index + model_name + '_' + dataset + '_' + job_id + '_' + str(node_size)
    print('log_dir', log_dir)
    if args.run_name is not None:
        log_dir += '_' + args.run_name
    if master_process:
        writer = SummaryWriter(log_dir)

    # ==============================
    # Initialize Model, Optimizer and LR Scheduler
    # ==============================
    config = XModelConfig.from_name(model_name)
    config.initializer_range = args.initializer_range
    print_rank_0(config)

    model = XModelForCausalLM(config)
    print_rank_0(model)

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        print_rank_0("Gradient-checkpointing enabled successfully")

    if args.use_flash_attn:
        # replace_with_flash_attention(model=model)
        print_rank_0("Flash-attention enabled successfully")

    total_params = sum(p.numel() for p in model.parameters())
    print_rank_0('Total params: ' + str(total_params))
    flops_per_token = model.flops_per_token(max_length=block_size)
    print_rank_0('flops per token: ' + str(flops_per_token))

    if master_process:
        print_model_ratio(model)

    model.to(device)
    # initialize a GradScaler. If enabled=False scaler is a no-op
    # scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    scaler = torch.cuda.amp.GradScaler(enabled=False)

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

    checkpoint = None  # free up memory

    # compile the model
    if compile:
        print_rank_0("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        # print('111_step_cuda_memory:{memo}G'.format(memo=torch.cuda.memory_allocated() / 1024 / 1024 / 1024))
        # print('111_step_max_cuda_memory:{memo}G'.format(memo=torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))
        model = DDP(model, device_ids=[ddp_local_rank])
        # print('222_step_cuda_memory:{memo}G'.format(memo=torch.cuda.memory_allocated() / 1024 / 1024 / 1024))
        # print('222_step_max_cuda_memory:{memo}G'.format(memo=torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))

    if args.noise_scale:
        bs_finder = BSFinder(model, bs=tokens_per_iter, num_it=args.total_steps, n_batch=20, beta=0.99)
        bs_finder.on_train_begin()

    # training loop
    data = get_batch('train')  # fetch the very first batch
    raw_model = model.module if ddp else model  # unwrap DDP container if needed
    running_mfu = -1.0
    dt_get_batch = float('inf')
    dt = float('inf')

    if master_process:
        t = trange(total_steps, desc='', leave=True)
    else:
        t = range(total_steps)
    for iter_num in t:
        # ti0=time.time()
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        num_tokens_trained = iter_num * tokens_per_iter

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            with catchtime() as elapsed_time:
                losses = estimate_loss()

            writer.add_scalar("Time/estimate_loss", elapsed_time(), num_tokens_trained)
            # print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            writer.add_scalar("Main/train_loss", losses['train'], num_tokens_trained)
            writer.add_scalar("Main/val_loss", losses['val'], num_tokens_trained)
            writer.add_scalar("Main/lr", lr, num_tokens_trained)
            writer.add_scalar("Main/mfu", running_mfu * 100, num_tokens_trained)
            writer.add_scalar("Main/throughput", tokens_per_iter / dt, num_tokens_trained)

            with catchtime() as elapsed_time:
                mae, mse = calculate_norm(model)

            writer.add_scalar("Time/calculate_norm", elapsed_time(), num_tokens_trained)
            writer.add_scalar("Norm/MAE", mae, num_tokens_trained)
            writer.add_scalar("Norm/MSE", mse, num_tokens_trained)

            with catchtime() as elapsed_time:
                l2_norm_dict = calculate_l2_norm(model)

            writer.add_scalar("Time/calculate_l2_norm", elapsed_time(), num_tokens_trained)
            for key in l2_norm_dict.keys():
                writer.add_scalar(f"Norm/l2 norm ({key})", l2_norm_dict[key], num_tokens_trained)

            with catchtime() as elapsed_time:
                sparsity_dict = calculate_sparsity(model)

            writer.add_scalar("Time/calculate_sparsity", elapsed_time(), num_tokens_trained)

            for key in sparsity_dict.keys():
                writer.add_scalar('Sparsity/' + key, sparsity_dict[key], num_tokens_trained)

            if args.calculate_ortho:
                with catchtime() as elapsed_time:
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
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    raw_model.save_pretrained(save_folder, is_main_process=master_process, safe_serialization=False)

        if iter_num == 0 and eval_only:
            break

        t0 = time.time()
        dt_get_batch = 0

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
                print('micro_step',micro_step)
                print('444_step_cuda_memory:{memo}G'.format(memo=torch.cuda.memory_allocated() / 1024 / 1024 / 1024))
                print('444_step_max_cuda_memory:{memo}G'.format(memo=torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))
                loss = model(data, labels=data).loss
                loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation
                print('555_step_cuda_memory:{memo}G'.format(memo=torch.cuda.memory_allocated() / 1024 / 1024 / 1024))
                print('555_step_max_cuda_memory:{memo}G'.format(
                    memo=torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))

            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            t0_get_batch = time.time()
            data = get_batch('train')
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
            dt_get_batch += time.time() - t0_get_batch
            print('666_step_cuda_memory:{memo}G'.format(memo=torch.cuda.memory_allocated() / 1024 / 1024 / 1024))
            print('666_step_max_cuda_memory:{memo}G'.format(memo=torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))

        if iter_num % eval_interval == 0 and master_process:
            # Compute the global L2 gradient norm
            with catchtime() as elapsed_time:
                grad_norm = calculate_grad_norm(model)

            writer.add_scalar("Time/calculate_grad_norm", elapsed_time(), num_tokens_trained)
            writer.add_scalar("Norm/grad norm", grad_norm, num_tokens_trained)

        if iter_num > 0 and master_process and args.noise_scale:
            # Compute the critical batch size
            with catchtime() as elapsed_time:
                result = bs_finder.on_backward_end(iter_num)

            if result:
                writer.add_scalar("Time/bs_finder.on_backward_end", elapsed_time(), num_tokens_trained)

                noise = result['noise']
                scale = result['scale']
                noise_scale = result['noise_scale']
                writer.add_scalar("Norm/noise", noise, num_tokens_trained)
                writer.add_scalar("Norm/scale", scale, num_tokens_trained)
                writer.add_scalar("Norm/noise scale", noise_scale, num_tokens_trained)

        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        print('777_step_cuda_memory:{memo}G'.format(memo=torch.cuda.memory_allocated() / 1024 / 1024 / 1024))
        print('777_step_max_cuda_memory:{memo}G'.format(memo=torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))
        scaler.update()
        print('888_step_cuda_memory:{memo}G'.format(memo=torch.cuda.memory_allocated() / 1024 / 1024 / 1024))
        print('888_step_max_cuda_memory:{memo}G'.format(memo=torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))

        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0

        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps

            bar_desc = f"loss: {lossf:.4f}"
            t.set_description(bar_desc, refresh=True)

            writer.add_scalar("Main/loss", lossf, num_tokens_trained)

            flops_achieved, mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt,
                                                         max_length=block_size, device_model='DCU')
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

            writer.add_scalar("Main/flops_achieved", flops_achieved*ddp_world_size, num_tokens_trained)
            writer.add_scalar("Main/mfu", running_mfu * 100, num_tokens_trained)
            writer.add_scalar("Main/throughput", tokens_per_iter / dt, num_tokens_trained)
            writer.add_scalar("Main/loss scale", scaler.get_scale(), num_tokens_trained)
            writer.add_scalar("Time/get batch", dt_get_batch, num_tokens_trained)

    if ddp:
        destroy_process_group()
