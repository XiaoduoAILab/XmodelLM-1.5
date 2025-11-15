import json
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from datasets import load_dataset, concatenate_datasets, Dataset

types = ['embed_tokens',
         'self_attn',
         'mlp',
         'lm_head',
         'self_attn.q_proj',
         'self_attn.k_proj',
         'self_attn.v_proj',
         'self_attn.o_proj',
         'mlp.gate_proj',
         'mlp.up_proj',
         'mlp.down_proj',
         'input_layernorm',
         'post_attention_layernorm',
         ]


def get_model_numel(model: nn.Module) -> int:
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def format_numel_str(numel: int) -> str:
    B = 1024 ** 3
    M = 1024 ** 2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"


def eval_ppl(dataset, model, tokenizer, device='cuda'):
    encodings = tokenizer("\n\n".join(dataset), return_tensors="pt")

    # max_length = model.config.n_positions
    max_length = 1024
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    latencies = []
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over input tokens.
            # Multiply it with trg_len to get the summation instead of average.
            # We will take average over all the tokens to get the true average
            # in the last step of this example.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    nlls = [num.item() for num in nlls]
    # print(nlls)
    avg_loss = sum(nlls) / end_loc
    # print('avg_loss: ' + str(avg_loss))
    ppl = np.exp(avg_loss)

    # ppl: tensor(25.1704, device='cuda:0')
    return ppl


def eval_lambada(model, tokenizer):
    data = load_dataset("text", data_files={"test": "data/lambada/lambada_test_plain_text.txt"}, split='test')['text']
    # print(data)
    # print('len(data): ' + str(len(data)))

    ppl = eval_ppl(data, model, tokenizer)

    return ppl


def eval_ptb(model, tokenizer):
    data = load_dataset("text", data_files={"test": "data/ptb/ptb.test.txt"}, split='test')['text']
    # print(data)
    # print('len(data): ' + str(len(data)))
    return eval_ppl(data, model, tokenizer)


def eval_wikitext2(model, tokenizer):
    # arrow_files = ["data/wikitext2/wikitext-train.arrow",
    #                "data/wikitext2/wikitext-validation.arrow",
    #                "data/wikitext2/wikitext-test.arrow", ]
    arrow_files = ["data/wikitext2/wikitext-test.arrow"]

    data = concatenate_datasets([Dataset.from_file(arrow_file) for arrow_file in arrow_files])["text"]
    # print(data)
    # print('len(data): ' + str(len(data)))

    return eval_ppl(data, model, tokenizer)


def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def calculate_ratio(model):
    total_params = sum(p.numel() for p in model.parameters())

    matrices = []
    for name, param in model.named_parameters():
        matrices.append(dict(name=name, num_params=param.numel()))

    types = ['embed_tokens', 'self_attn', 'mlp', 'lm_head', 'input_layernorm', 'post_attention_layernorm', ]
    data = dict()
    for t in types:
        data[t] = [0]

    for item in matrices:
        name = item['name']
        num_params = item['num_params']

        for t in types:
            if t in name:
                data[t][0] += num_params

    result = dict()
    for t in types:
        result[t] = data[t][0] / total_params

    return result


def calculate_norm(model):
    total_params = sum(p.numel() for p in model.parameters())

    total_norm = 0
    average_abs = 0
    for p in [p for p in model.parameters() if p.requires_grad]:
        total_norm += p.data.norm(2).item() ** 2
        average_abs += p.numel() / total_params * p.data.abs().mean().item()

    average_norm = (total_norm / total_params) ** 0.5
    return average_abs, average_norm


def calculate_l2_norm(model):
    result = dict()
    for t in types:
        result[t] = 0

    for name, param in model.named_parameters():
        for t in types:
            if t in name:
                result[t] += param.data.norm(2).item() ** 2

    for t in types:
        result[t] = result[t] ** 0.5
    return result


def calculate_grad_norm(model):
    grads = [
        param.grad.detach().flatten()
        for param in model.parameters()
        if param.grad is not None
    ]
    norm = torch.cat(grads).norm()
    return norm.float()


def calculate_sparsity(model):
    total_params = sum(p.numel() for p in model.parameters())

    matrices = []
    for name, param in model.named_parameters():
        # print(name, param.data.shape)
        W = param.data.double()
        l1_norm = torch.norm(W, p=1).cpu().item()
        # print('l1_norm: ' + str(l1_norm))
        l2_norm = torch.norm(W, p=2).cpu().item()
        # print('l2_norm: ' + str(l2_norm))
        sparsity = l1_norm / l2_norm / np.sqrt(param.numel())
        # print(mse)
        matrices.append(dict(name=name, sparsity=sparsity, num_params=param.numel()))

    data = dict()
    for t in types:
        data[t] = [0, 0]

    for item in matrices:
        name = item['name']
        sparsity = item['sparsity']
        num_params = item['num_params']

        for t in types:
            if t in name:
                data[t][0] += num_params
                data[t][1] += sparsity * num_params

    result = dict()
    for t in types:
        result[t] = data[t][1] / total_params

    return result


def calculate_orthog_mse(model):
    total_params = sum(p.numel() for p in model.parameters())

    matrices = []
    for name, param in model.named_parameters():
        # print(name, param.data.shape)
        n = param.data.shape[-1]
        W = param.data
        W_T = W.permute(*torch.arange(W.ndim - 1, -1, -1))
        X = torch.matmul(W_T.type(torch.DoubleTensor), W.type(torch.DoubleTensor)).cpu().numpy()
        mse = np.square(X - np.eye(n)).mean()
        # print(mse)
        matrices.append(dict(name=name, mse=mse, num_params=param.numel()))

    data = dict()
    for t in types:
        data[t] = [0, 0]

    for item in matrices:
        name = item['name']
        mse = item['mse']
        num_params = item['num_params']

        total_params += num_params

        for t in types:
            if t in name:
                data[t][0] += num_params
                data[t][1] += mse * num_params

    result = dict()
    for t in types:
        result[t] = data[t][1] / total_params

    return result


def print_model_ratio(model):
    ratio_dict = calculate_ratio(model)
    total_ratio = 0
    print('\nratio:')
    for key in ratio_dict.keys():
        total_ratio += ratio_dict[key]
        print('    {}: {:.8f}'.format(key, ratio_dict[key]))

    print('    total: ' + str(total_ratio))


def extract_embeddings(model, tokenizer, texts):
    model.eval()

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        last_hidden_state = model(**inputs, output_hidden_states=True).hidden_states[-1]

    weights_for_non_padding = inputs.attention_mask * torch.arange(start=1,
                                                                   end=last_hidden_state.shape[1] + 1).unsqueeze(0)

    sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
    num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
    sentence_embeddings = sum_embeddings / num_of_none_padding_tokens

    return sentence_embeddings


def extract_embeddings_prompt_eol(model, tokenizer, texts):
    # https://stackoverflow.com/questions/76926025/sentence-embeddings-from-llama-2-huggingface-opensource
    model.eval()

    texts = [f"This sentence: {text} means in one word:" for text in texts]

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        last_hidden_state = model(**inputs, output_hidden_states=True).hidden_states[-1]

    weights_for_non_padding = inputs.attention_mask * torch.arange(start=1,
                                                                   end=last_hidden_state.shape[1] + 1).unsqueeze(0)

    sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
    num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
    sentence_embeddings = sum_embeddings / num_of_none_padding_tokens

    return sentence_embeddings


def choose_last_iter(save_folder):
    folders = [f for f in os.listdir(save_folder) if
               os.path.isdir(os.path.join(save_folder, f)) and f.startswith('iter-')]
    return sorted(folders)[-1]


def load_jsonl(file_path):
    result = []
    with open(file_path, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        item = json.loads(line.strip())
        result.append(item)

    return result


def flops_calculator(config, batch_size, seq_len):
    checkpoint_activations_factor = 3
    flops_per_iteration = (24 * checkpoint_activations_factor * batch_size * seq_len * config.num_hidden_layers * (
            config.hidden_size ** 2)) * (1. + (seq_len / (6. * config.hidden_size)) + (
            config.vocab_size / (16. * config.num_hidden_layers * config.hidden_size)))
    return flops_per_iteration


def estimate_mfu(config, total_params, fwdbwd_per_iter, dt, max_length=None, device_model='A100', dtype='float32'):
    """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
    # first estimate the number of flops we do per iteration.
    # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    N = total_params
    n_layer = config.num_hidden_layers
    n_head = config.num_attention_heads
    n_embd = config.hidden_size

    if max_length is None:
        max_length = config.max_position_embeddings

    L, H, Q, T = n_layer, n_head, n_embd // n_head, max_length
    flops_per_token = 6 * N + 12 * L * H * Q * T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    # express our flops throughput as ratio of A100 bfloat16 peak flops
    flops_achieved = flops_per_iter * (1.0 / dt)  # per second

    if device_model is None:
        device_model = torch.cuda.get_device_name(0)

    flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
    if device_model == 'DCU' and dtype == 'float16':
        flops_promised = 23.6e12
    elif device_model == 'DCU' and dtype == 'float32':
        flops_promised = 11.8e12
    elif device_model == 'NVIDIA V100' or 'V100' in device_model:
        flops_promised = 28e12
    elif device_model == 'NVIDIA H100' or 'H100' in device_model or device_model == 'NVIDIA H800' or 'H800' in device_model:
        flops_promised = 1513e12

    mfu = flops_achieved / flops_promised
    return flops_achieved, mfu


def flops_per_token(config, num_params, max_length=None, non_embedding=False):
    N = num_params
    if non_embedding:
        N -= config.vocab_size * config.hidden_size * 2
    n_layer = config.num_hidden_layers
    n_head = config.num_attention_heads
    n_embd = config.hidden_size
    if max_length is None:
        max_length = config.max_position_embeddings
    L, H, Q, T = n_layer, n_head, n_embd // n_head, max_length
    flops_per_token = 6 * N + 12 * L * H * Q * T
    return flops_per_token


# def reset_rope(model, model_max_train_len, scaling_factor):
#     for l in model.model.layers:
#         if hasattr(l.self_attn, 'rotary_emb'):
#             l.self_attn.rotary_emb.scaling_factor = scaling_factor
#             try:
#                 l.self_attn.rotary_emb._set_cos_sin_cache(seq_len=model_max_train_len, device="cpu",
#                                                           dtype=torch.float32)
#             except AttributeError:
#                 print(
#                     "Error: _set_cos_sin_cache method not found. Consider downgrading transformers to version 4.35.2.")
#
#     return

# This is the continue pretrained LLaMA 2 7B model with modified rope
# AttributeError: 'LlamaRotaryEmbedding' object has no attribute '_set_cos_sin_cache'
# Downgrading to 4.37.2 worked for me
# pip install transformers==4.37.2
def reset_rope(model, model_max_train_len, scaling_factor):
    for l in model.model.layers:
        l.self_attn.rotary_emb.scaling_factor = scaling_factor
        l.self_attn.rotary_emb._set_cos_sin_cache(seq_len=model_max_train_len, device=model.device, dtype=torch.float32)
    return
