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
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import deepspeed
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from models.modeling_xmodel import XModelForCausalLM
from models.configuration_xmodel import XModelConfig
from utils import setup_seeds, calculate_norm, calculate_sparsity, calculate_orthog_mse, \
    calculate_l2_norm, print_model_ratio

from data_utils import create_dataloaders
root_path=os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
from Pytorch_Memory_Utils.gpu_mem_track import MemTracker
# from pytorch_memlab import MemReporter
from memory_profiler import profile

def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=-1, help="Reserved for deepspeed framework")
    parser.add_argument("--model_name", type=str, default='medium', help="model name")
    parser.add_argument("--run_name", type=str, default=None, help="run name")
    parser.add_argument("--data_path", type=str,
                        default=root_path+"/data/openwebtext/",
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
    parser.add_argument("--job_id", type=str, default='0', help="slurm job id")
    parser.add_argument("--node_size", type=int, default=1, help="parallel node number")
    parser.add_argument("--line_index", type=str, default='line/', help="tensorbord log data in which line")

    return parser


def get_batch():
    return next(dataloader_iter).to(device_flag)


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

@profile
def train(iter_num):
    global batch, dt_get_batch, dt
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    num_tokens_trained = iter_num * tokens_per_iter

    if iter_num % eval_interval == 0 and master_process:
        # with catch_time() as elapsed_time:
        #     mae, mse = calculate_norm(model)
        #
        # writer.add_scalar("Time/calculate_norm", elapsed_time(), num_tokens_trained)
        # writer.add_scalar("Norm/MAE", mae, num_tokens_trained)
        # writer.add_scalar("Norm/MSE", mse, num_tokens_trained)
        #
        # with catch_time() as elapsed_time:
        #     l2_norm_dict = calculate_l2_norm(model)
        #
        # writer.add_scalar("Time/calculate_l2_norm", elapsed_time(), num_tokens_trained)
        # for key in l2_norm_dict.keys():
        #     writer.add_scalar(f"Norm/l2 norm ({key})", l2_norm_dict[key], num_tokens_trained)
        #
        # with catch_time() as elapsed_time:
        #     sparsity_dict = calculate_sparsity(model)
        #
        # writer.add_scalar("Time/calculate_sparsity", elapsed_time(), num_tokens_trained)
        #
        # for key in sparsity_dict.keys():
        #     writer.add_scalar('Sparsity/' + key, sparsity_dict[key], num_tokens_trained)
        #
        # if args.calculate_ortho:
        #     with catch_time() as elapsed_time:
        #         ortho_mse_dict = calculate_orthog_mse(model)
        #
        #     writer.add_scalar("Time/calculate_ortho_mse", elapsed_time(), num_tokens_trained)
        #
        #     for key in ortho_mse_dict.keys():
        #         writer.add_scalar('Orthogonality/' + key, ortho_mse_dict[key], num_tokens_trained)
        print('iter_num',iter_num)
        if always_save_checkpoint:
            if iter_num > 0:
                save_folder = os.path.join(out_dir, model_name)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                save_folder = os.path.join(save_folder, f"iter-{iter_num:06d}")
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                print(f"saving checkpoint to {save_folder}")

                # model_engine.save_checkpoint(save_dir=save_folder)
                # print('model_engine',model_engine)
                # print('model_engine_',model_engine.modules())
                # model.save_pretrained(save_folder, is_main_process=master_process, safe_serialization=False)
                # print('optimizer',model_engine.optimizer.state_dict())
                # checkpoint = {
                #     'optimizer': model_engine.optimizer.state_dict(),
                #     'model_args': vars(args),
                #     'iter_num': iter_num,
                #     'best_val_loss': best_val_loss,
                #     'config': config,
                # }
                # torch.save(checkpoint, os.path.join(save_folder, 'ckpt.pt'))

    t0 = time.time()
    dt_get_batch = 0
    print('222_step_cuda_memory:{memo:.2f}G'.format(memo=torch.cuda.memory_allocated() / 1024 / 1024 / 1024))
    print('222_step_max_cuda_memory:{memo:.2f}G'.format(memo=torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    for micro_step in range(gradient_accumulation_steps):
        # backward pass
        print('micro_step',micro_step)
        print('331_step_cuda_memory:{memo:.2f}G'.format(memo=torch.cuda.memory_allocated() / 1024 / 1024 / 1024))
        # gpu_tracker.track()
        loss = model_engine(batch, labels=batch).loss
        loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation
        # weight update
        # _global_grad_norm=model_engine.get_global_grad_norm()
        # print('_global_grad_norm',_global_grad_norm)
        # gpu_tracker.track()
        print('332_step_cuda_memory:{memo:.2f}G'.format(memo=torch.cuda.memory_allocated() / 1024 / 1024 / 1024))
        # gpu_tracker.track()
        # backward pass
        model_engine.backward(loss)
        print('333_step_cuda_memory:{memo:.2f}G'.format(memo=torch.cuda.memory_allocated() / 1024 / 1024 / 1024))
        print('333_step_max_cuda_memory:{memo:.2f}G'.format(memo=torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))
        # gpu_tracker.track()
        model_engine.step()

        t0_get_batch = time.time()
        batch = get_batch()
        dt_get_batch += time.time() - t0_get_batch

    t0_get_batch = time.time()
    batch = get_batch()
    dt_get_batch += time.time() - t0_get_batch
    # if iter_num % eval_interval == 0 and master_process:
    #     # Compute the global L2 gradient norm
    #     with catch_time() as elapsed_time:
    #         grad_norm = calculate_grad_norm(model)
    #
    #     writer.add_scalar("Time/calculate_grad_norm", elapsed_time(), num_tokens_trained)
    #     writer.add_scalar("Norm/grad norm", grad_norm, num_tokens_trained)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    # reporter.report()
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps

        bar_desc = f"loss: {lossf:.4f}"
        t.set_description(bar_desc, refresh=True)

        writer.add_scalar("Main/loss", lossf, num_tokens_trained)

        flops_achieved, mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt,
                                                 max_length=max_length, device_type=args.device_model,
                                                 dtype=dtype)

        writer.add_scalar("Main/flops_achieved", flops_achieved*ddp_world_size, num_tokens_trained)
        writer.add_scalar("Main/mfu", mfu * 100, num_tokens_trained)
        writer.add_scalar("Main/throughput", tokens_per_iter / dt, num_tokens_trained)
        writer.add_scalar("Main/loss", lossf, num_tokens_trained)

        writer.add_scalar("Time/get batch", dt_get_batch, num_tokens_trained)
        # writer.add_scalar("Time/cuda activat memory", round(torch.cuda.memory_allocated() / 1024 / 1024 / 1024,3),num_tokens_trained)


if __name__ == "__main__":
    setup_seeds(42)
    # gpu_tracker = MemTracker()
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

    print('OMPI_COMM_WORLD_LOCAL_RANK', os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK'), os.environ.get('LOCAL_RANK'))
    os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = os.environ.get('LOCAL_RANK')
    world_rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
    ddp_rank = world_rank % torch.cuda.device_count()
    device_flag = f'cuda:{ddp_rank}'
    torch.cuda.set_device(device_flag)
    args.local_rank = ddp_rank
    master_process = world_rank == 0
    seed_offset = world_rank
    # print('args_local_rank_master_process222', args.local_rank, master_process)
    # print('world_rank_world_size_ddp_rank', world_rank, world_size, ddp_rank)

    ds_config = json.load(open(f'config/deepspeed_{model_name}.json'))
    ds_config['gradient_accumulation_steps'] = gradient_accumulation_steps
    ds_config['train_micro_batch_size_per_gpu'] = batch_size


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
    # data_config = [
    #     ("arxiv", 2.5),
    #     ("book", 4.5),
    #     ("c4", 15.0),
    #     ("cc", 67.0),
    #     ("github", 4.5),
    #     ("stackexchange", 2.0),
    #     ("wikipedia", 4.5),
    # ]
    data_config = [("train", 100)]
    eval_data_config = [("val", 100)]
    print('args.data_path',args.data_path)
    train_dataloader, _ = create_dataloaders(
        batch_size=args.micro_batch_size,
        block_size=args.max_length,
        data_config=data_config,
        sampling=args.sampling,
        train_data_dir=args.data_path,
        val_data_dir=None,
        seed=1337 + seed_offset
    )
    _, val_dataloader = create_dataloaders(
        batch_size=args.micro_batch_size,
        block_size=args.max_length,
        data_config=data_config,
        sampling=args.sampling,
        train_data_dir=args.data_path,
        val_data_dir=None,
        seed=1337 + seed_offset
    )

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # init a new model from scratch
    print_rank_0("Initializing a new model from scratch")

    # logging
    # ds_logger.setLevel(logging.DEBUG)
    log_dir = 'runs/' + line_index + model_name + '_' + dataset + '_' + job_id + '_' + str(node_size)
    if args.run_name is not None:
        log_dir += '_' + args.run_name

    if master_process:
        writer = SummaryWriter(log_dir)

    # ==============================
    # Initialize Model, Optimizer and LR Scheduler
    # ==============================
    # gpu_tracker.track()
    config = XModelConfig.from_name(model_name)
    print_rank_0(config)

    model = XModelForCausalLM(config)
    print_rank_0(model)
    # gpu_tracker.track()

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        print_rank_0("Gradient-checkpointing enabled successfully")

    if args.use_flash_attn:
        print('use_flash_attn', args.use_flash_attn)
        # from patch.flash_attention_patch import replace_with_flash_attention
        # replace_with_flash_attention(model=model)
        print_rank_0("Flash-attention enabled successfully")

    total_params = sum(p.numel() for p in model.parameters())
    print_rank_0('Total params: ' + str(total_params))
    flops_per_token = model.flops_per_token(max_length=max_length)
    print_rank_0('flops per token: ' + str(flops_per_token))

    if master_process:
        print('000_step_cuda_memory:{memo:.2f}G'.format(memo=torch.cuda.memory_allocated() / 1024 / 1024 / 1024))
        print('000_step_max_cuda_memory:{memo:.2f}G'.format(memo=torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))
        print_model_ratio(model)

    model.to(device_flag)
    # gpu_tracker.track()

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
    model_engine, _, _, _ = deepspeed.initialize(args=args,
                                                 model=model,
                                                 # optimizer=optimizer,
                                                 model_parameters=model.parameters(),
                                                 config=ds_config)
    # gpu_tracker.track()

    model_engine.train()
    # reporter=MemReporter(model_engine)
    print('001_step_cuda_memory:{memo:.2f}G'.format(memo=torch.cuda.memory_allocated() / 1024 / 1024 / 1024))
    print('001_step_max_cuda_memory:{memo:.2f}G'.format(memo=torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))
    # training loop
    dataloader_iter = iter(train_dataloader)
    batch = get_batch()  # fetch the very first batch

    dt_get_batch = float('inf')
    dt = float('inf')

    if master_process:
        print('111_step_cuda_memory:{memo:.2f}G'.format(memo=torch.cuda.memory_allocated() / 1024 / 1024 / 1024))
        print('111_step_max_cuda_memory:{memo:.2f}G'.format(memo=torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))
        t = trange(total_steps, desc='', leave=True)
    else:
        t = range(total_steps)

    # prof = torch.profiler.profile(
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
    #     record_shapes=True,
    #     with_stack=True)
    # prof.start()
    # gpu_tracker.track()
    for iter_num in t:
        train(iter_num)
        # prof.step()

    # prof.stop()
