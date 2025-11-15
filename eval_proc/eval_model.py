import argparse
import json
import os
import sys
from pathlib import Path

import torch
import transformers
from datasets import load_dataset, Dataset

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from models.modeling_xmodel import XModelForCausalLM
from models.configuration_xmodel import XModelConfig
from utils import deploy_model, eval_ppl, eval_acc
from winograd import eval_winograd
from wmt import eval_wmt
from coqa import eval_coqa
from Mteb import eval_mteb_cls, eval_mteb_sts

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# export HF_ENDPOINT=https://hf-mirror.com


ds_skip = ['lambada', 'ptb', 'wikitext', 'winograd', 'wmt', 'coqa', 'mteb_cls', 'mteb_sts']
ds_skip = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, default="/data4/liuyang/checkpoints/c_line/medium/iter-350000")
    parser.add_argument("--model_path", type=str, default="/data/wangyichuan/train/out/nano_lsh_3/iter-058000")
    parser.add_argument("--model_name", type=str, default="nano")
    parser.add_argument("--device", type=str, default="cuda:7")
    args = parser.parse_args()

    device = args.device
    model_path, model_name = args.model_path, args.model_name
    print('model_name: ' + model_path)

    # load data
    lambada = load_dataset("text", data_files={"test": "/data/wangyichuan/llm_pretrain_xmodel/data/lambada/lambada_test_plain_text.txt"}, split='test')[
        'text']
    ptb = load_dataset("text", data_files={"test": "/data/wangyichuan/llm_pretrain_xmodel/data/ptb/ptb.test.txt"}, split='test')['text']
    wikitext = Dataset.from_file('/data/wangyichuan/llm_pretrain_xmodel/data/wikitext2/wikitext-test.arrow')['text']

    datasets = {'lambada': lambada, 'ptb': ptb, 'wikitext': wikitext, 'lambada_acc': lambada, 'ptb_acc': ptb, 'wikitext_acc': wikitext, 'winograd': 1, 'wmt': 1, 'coqa': 1, 'mteb_cls': 1, 'mteb_sts': 1}

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained('/data/wangyichuan/llm_pretrain_xmodel/tokenizers/llama', use_fast=False, trust_remote_code=True)
    print(tokenizer)

    config = XModelConfig.from_name(model_name)
    model = XModelForCausalLM(config)

    model.load_state_dict(torch.load('%s/pytorch_model.bin'%model_path))
    model.to(device)
    model.eval()

    with open(f'eval_{model_name}.jsonl', 'a') as fp:

        for eval_name in datasets.keys():

            if eval_name in ds_skip:
                continue

            if eval_name in ['lambada', 'ptb', 'wikitext']:
                data = datasets[eval_name]
                ppl = eval_ppl(data, model, tokenizer, device)
                # print('ppl: ' + str(ppl))
                line = json.dumps(dict(eval_name=eval_name, ppl=ppl))
                print(line)
                fp.write(line + '\n')
            
            elif eval_name in ['lambada_acc', 'ptb_acc', 'wikitext_acc']:
                data = datasets[eval_name]
                acc = eval_acc(data, model, tokenizer, device)
                # print('ppl: ' + str(ppl))
                line = json.dumps(dict(eval_name=eval_name, acc=acc))
                print(line)
                fp.write(line + '\n')
            
            elif eval_name in ['winograd']:
                acc = eval_winograd(model, tokenizer, device)
                line = json.dumps(dict(eval_name=eval_name, acc=acc))
                print(line)
                fp.write(line + '\n')
            
            elif eval_name in ['wmt']:
                bleu = eval_wmt(model, tokenizer, device)
                line = json.dumps(dict(eval_name=eval_name, bleu=bleu))
                print(line)
                fp.write(line + '\n')
            
            elif eval_name in ['coqa']:
                f1 = eval_coqa(model, tokenizer, device)
                line = json.dumps(dict(eval_name=eval_name, f1=f1))
                print(line)
                fp.write(line + '\n')
            
            elif eval_name in ['mteb_cls']:
                f1 = eval_mteb_cls(model, tokenizer)
                line = json.dumps(dict(eval_name=eval_name, f1=f1))
                print(line)
                fp.write(line + '\n')
            
            elif eval_name in ['mteb_sts']:
                sim = eval_mteb_sts(model, tokenizer)
                line = json.dumps(dict(eval_name=eval_name, sim=sim))
                print(line)
                fp.write(line + '\n')