import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import transformers
from datasets import load_dataset, Dataset

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from models.modeling_xmodel import XModelForCausalLM
from models.configuration_xmodel import XModelConfig
from utils import deploy_model, eval_ppl, eval_acc
from Mteb import eval_mteb_cls, eval_mteb_sts

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# export HF_ENDPOINT=https://hf-mirror.com

ds_skip = ['lambada', 'ptb', 'wikitext', 'lambada_acc', 'ptb_acc', 'wikitext_acc', 'arc_challenge', 'arc_easy', 'boolq',
           'openbookqa', 'piqa', 'winogrande', 'mteb_cls', 'mteb_sts']
ds_skip = ['lambada', 'ptb', 'wikitext', 'lambada_acc', 'ptb_acc', 'wikitext_acc', 'mteb_cls', 'mteb_sts']


def do_task(task, fp):
    start = time.time()
    command = f"lm_eval --model hf --model_args pretrained=working_dir_%s,trust_remote_code=True --tasks {task} --device %s --batch_size %d" % (
        args.model_name, device, args.batch_size)

    result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)

    match = re.search(r"\|acc\s*\|\s*([\d.]+)\s*\|", result.stdout)
    acc = float(match.group(1))
    line = json.dumps(dict(ckp_name=ckp_name, acc=acc))
    print(line, 'time: ', time.time() - start)
    fp.write(line + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=str, default="/data4/liuyang/checkpoints/c_line")
    parser.add_argument("--model_name", type=str, default="medium")
    parser.add_argument("--device", type=str, default="cuda:6")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = args.device
    model_name = args.model_name
    model_path = os.path.join(args.checkpoints_dir, args.model_name)
    print('model_name: ' + str(model_name))

    # load data
    lambada = load_dataset("text", data_files={
        "test": "data/lambada/lambada_test_plain_text.txt"}, split='test')[
        'text']
    ptb = load_dataset("text", data_files={"test": "data/ptb/ptb.test.txt"},
                       split='test')['text']
    wikitext = Dataset.from_file('data/wikitext2/wikitext-test.arrow')['text']

    datasets = {'lambada': lambada, 'ptb': ptb, 'wikitext': wikitext, 'lambada_acc': lambada, 'ptb_acc': ptb,
                'wikitext_acc': wikitext, 'arc_challenge': 1, 'arc_easy': 1, 'boolq': 1, 'openbookqa': 1, 'piqa': 1,
                'winogrande': 1, 'mteb_cls': 1, 'mteb_sts': 1}

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained('tokenizers/llama',
                                                           use_fast=False, trust_remote_code=True)
    print(tokenizer)

    dirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
    dirs = sorted(dirs)
    # dirs = [dirs[i*5] for i in range(len(dirs)//5)]
    print(len(dirs))

    for ckp_name in dirs:

        # if int(ckp_name[-6:]) < 98000:
        #     continue

        ckp_folder = os.path.join(model_path, ckp_name)
        deploy_model(ckp_folder, model_name)

        config = XModelConfig.from_name(model_name)
        model = XModelForCausalLM(config)
        PATH = 'working_dir_%s/pytorch_model.bin' % model_name
        model.load_state_dict(torch.load(PATH))

        model.to(device)
        model.eval()

        for eval_name in datasets.keys():

            if eval_name in ds_skip:
                continue

            if not os.path.exists(f'eval_result/{model_name}'):
                os.makedirs(f'eval_result/{model_name}', exist_ok=True)

            with open(f'eval_result/{model_name}/{eval_name}_{model_name}.jsonl', 'a+') as fp:
                print(f'eval_{eval_name}_{model_name} <- {ckp_name}')

                if eval_name in ['lambada', 'ptb', 'wikitext']:
                    data = datasets[eval_name]
                    ppl = eval_ppl(data, model, tokenizer, device)
                    line = json.dumps(dict(ckp_name=ckp_name, ppl=ppl))
                    print(line)
                    fp.write(line + '\n')

                elif eval_name in ['lambada_acc', 'ptb_acc', 'wikitext_acc']:
                    data = datasets[eval_name]
                    acc = eval_acc(data, model, tokenizer, device)
                    line = json.dumps(dict(ckp_name=ckp_name, acc=acc))
                    print(line)
                    fp.write(line + '\n')

                elif eval_name in ['arc_challenge']:
                    do_task(eval_name, fp)

                elif eval_name in ['arc_easy']:
                    do_task(eval_name, fp)

                elif eval_name in ['boolq']:
                    do_task(eval_name, fp)

                elif eval_name in ['openbookqa']:
                    do_task(eval_name, fp)

                elif eval_name in ['piqa']:
                    do_task(eval_name, fp)

                elif eval_name in ['winogrande']:
                    do_task(eval_name, fp)

                elif eval_name in ['mteb_cls']:
                    start = time.time()
                    f1 = eval_mteb_cls(model, tokenizer)
                    line = json.dumps(dict(ckp_name=ckp_name, f1=f1))
                    print(line, 'time: ', time.time() - start)
                    fp.write(line + '\n')

                elif eval_name in ['mteb_sts']:
                    start = time.time()
                    sim = eval_mteb_sts(model, tokenizer)
                    line = json.dumps(dict(ckp_name=ckp_name, sim=sim))
                    print(line, 'time: ', time.time() - start)
                    fp.write(line + '\n')

        shutil.rmtree('working_dir_%s' % model_name)
