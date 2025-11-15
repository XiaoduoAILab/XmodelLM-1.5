import argparse
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM


def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default='/data1/liuyang/checkpoints/e_line/xl/iter-0450000/')
    parser.add_argument("--output_path", type=str, default='/data1/liuyang/e_line_xl_iter_0450000_llama_format/')
    return parser


if __name__ == "__main__":
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    temp_folder = './tmp'

    config_src = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    print(config_src)

    file_path = os.path.join(args.model_path, 'pytorch_model.bin')
    if not os.path.isfile(file_path):
        model_src = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
        model_src.save_pretrained(temp_folder, safe_serialization=False)
        file_path = os.path.join(temp_folder, 'pytorch_model.bin')

    # param_src = dict()
    # for name, param in model_src.named_parameters():
    #     print(name)
    #     param_src[name] = param

    # print(model_src)

    config_dst = LlamaConfig()
    config_dst.__dict__.update(config_src.__dict__)
    # config_dst.bos_token_id = config_src.bos_token_id
    # config_dst.eos_token_id = config_src.eos_token_id
    # config_dst.hidden_act = config_src.hidden_act
    # config_dst.hidden_size = config_src.hidden_size
    # config_dst.intermediate_size = config_src.intermediate_size
    # config_dst.max_position_embeddings = config_src.max_position_embeddings
    # config_dst.num_attention_heads = config_src.num_attention_heads
    # config_dst.num_hidden_layers = config_src.num_hidden_layers
    # config_dst.num_key_value_heads = config_src.num_key_value_heads
    # config_dst.torch_dtype = config_src.torch_dtype
    # config_dst.rms_norm_eps = config_src.rms_norm_eps
    # config_dst.vocab_size = config_src.vocab_size

    config_dst.head_dim = config_dst.hidden_size // config_dst.num_heads

    print(config_dst)
    config_dst.save_pretrained(args.output_path)

    model_dst = LlamaForCausalLM(config_dst)

    model_dst.load_state_dict(torch.load(file_path), strict=False)

    # for name, param in model_dst.named_parameters():
    #     print(name)

    model_dst.save_pretrained(args.output_path, safe_serialization=False)
    print(model_dst)
