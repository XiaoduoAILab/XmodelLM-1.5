import argparse
import json
import os
import sys
from pathlib import Path

import transformers
from tqdm import tqdm

wd = Path(__file__).parent.parent.parent.resolve()
print('wd: ' + str(wd))
sys.path.append(str(wd))

from evaluate.utils import deploy_model


def calculate_l2_norm(model):
    result = 0

    for name, param in model.named_parameters():
        result += param.data.norm(2).item() ** 2

    result = result ** 0.5
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/data4/liuyang/checkpoints/e_line/")
    parser.add_argument("--model_name", type=str, default='xl', help="model name")
    args = parser.parse_args()

    model_dir = os.path.join(args.model_dir, args.model_name)
    tokenizer_path = f'tokenizers/xmodel/v6/xmodel_32000/'
    model_path = f'working_dir_{args.model_name}'

    dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    dirs = sorted(dirs)

    filename = f'{args.model_name}.jsonl'

    with open(filename, 'w') as fp:
        for ckp_name in tqdm(dirs):
            # print('ckp_name: ' + str(ckp_name))
            ckp_folder = os.path.join(model_dir, ckp_name)
            deploy_model(ckp_folder, args.model_name, tokenizer_path=tokenizer_path)

            model = transformers.AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

            l2_norm = calculate_l2_norm(model)

            log_line = json.dumps(dict(ckp_name=ckp_name, l2_norm=l2_norm)) + '\n'
            fp.write(log_line)
            fp.flush()
