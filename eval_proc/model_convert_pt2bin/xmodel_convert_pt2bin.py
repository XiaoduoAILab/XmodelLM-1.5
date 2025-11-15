from pathlib import Path
import glob, os, sys
from torch.utils.data import IterableDataset
import numpy as np
import torch
import pdb
import argparse
import deepspeed
import collections

models_root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(models_root_path)
from models.configuration_xmodel import XModelConfig
from models.modeling_xmodel import XModelForCausalLM


def substring_between(str, start, end):
    start_index = str.find(start)
    if start_index == -1:
        return ""  # 返回空字符串表示未找到起始字符串
    start_index += len(start)  # 跳过起始字符串本身

    end_index = str.find(end, start_index)
    if end_index == -1:
        return ""  # 返回空字符串表示未找到结束字符串

    return str[start_index:end_index]


def reload_megatron_model_dict(model_path):
    model_path_list = os.listdir(model_path)
    model_layer_list = [file_name for file_name in model_path_list if 'model_00-model_states.pt' in file_name]
    model_layer_list.sort(key=lambda x: int(x[len('layer_'):-len('-model_00-model_states.pt')]))
    model_state_dict_md = collections.OrderedDict()
    for file_name in model_layer_list:
        sub_dict = torch.load(model_path + '/' + file_name)
        sub_ordered_dict = collections.OrderedDict()
        for layer_name, layer_weight in sub_dict.items():
            # print('debug_layer_name', layer_name)
            if 'decoder_layer' in layer_name:
                new_layer_number = substring_between(file_name, 'layer_', '-model_00-model_states.pt')
                while new_layer_number[0] == '0': new_layer_number = new_layer_number[1:]
                new_layer_number = str(int(new_layer_number) - 2)
                new_layer_post_name = layer_name.replace('decoder_layer', '')
                new_layer_name = 'model.layers.' + new_layer_number + new_layer_post_name
                sub_ordered_dict[new_layer_name] = layer_weight
            elif 'lm_head.weight' in layer_name:
                sub_ordered_dict[layer_name] = layer_weight
            elif 'weight' == layer_name:
                pass
            else:  # norm.weight
                new_layer_name = 'model.' + layer_name
                sub_ordered_dict[new_layer_name] = layer_weight

        model_state_dict_md.update(sub_ordered_dict)

    return model_state_dict_md


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=models_root_path + "/out/large_megatron/global_step100")
    parser.add_argument("--model_path1", type=str,
                        default=models_root_path + "/out/large_megatron_official/global_step100")
    parser.add_argument("--model_path2", type=str, default="/data4/liuyang/checkpoints/a_line/nano/iter-009900")
    parser.add_argument("--model_name", type=str, default="micro")
    parser.add_argument("--device", type=str, default="cuda:6")
    args = parser.parse_args()
    device = args.device
    model_path = args.model_path
    model_path2 = args.model_path2

    model_state_dict_md = reload_megatron_model_dict(model_path)
    model_name = 'nano'
    config = XModelConfig.from_name(model_name)
    model = XModelForCausalLM(config)
    model_torch_version = XModelForCausalLM(config)
    print('%s/pytorch_model.bin' % model_path2)
    mode_parameter_dict = torch.load('%s/pytorch_model.bin' % model_path2)
    model.load_state_dict(mode_parameter_dict)
    model_torch_version.load_state_dict(model_state_dict_md)
    inputs = torch.tensor([[1, 2, 3, 4]])
    out = model.generate(inputs)
    out222 = model_torch_version.generate(inputs)
    out_dict = model(inputs)
    out_dict_torch_version = model_torch_version(inputs)

    print('111')
