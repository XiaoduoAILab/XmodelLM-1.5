import argparse
import json
import torch
import transformers
from tqdm import tqdm
from utils import eval_ppl_sent

import sys
from pathlib import Path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from models.modeling_xmodel import XModelForCausalLM
from models.configuration_xmodel import XModelConfig


def eval_winograd(model, tokenizer, device):
    num, num_acc = 0, 0
    with open(str(wd)+'/data/winograd/dev.jsonl') as f:
        for i in tqdm(range(1267)):
            line = f.readline()
            js = json.loads(line)
            text1, text2 = js['sentence'].replace('_',js['option1']), js['sentence'].replace('_',js['option2'])
            # text1, text2 = js['sentence'] + ' _ should be %s.'%js['option1'], js['sentence'] + ' _ should be %s'%js['option2']
            ppl1, ppl2 = eval_ppl_sent(text1, model, tokenizer, device), eval_ppl_sent(text2, model, tokenizer, device)
            pred, answer = 1*(ppl1<ppl2) + 2*(ppl1>ppl2), int(js['answer'])
            num += 1
            num_acc += 1*(pred==answer)
    return num_acc / num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:7", help="")
    args = parser.parse_args()

    model_dir = 'working_dir'
    model_name = 'small'
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(wd)+'/tokenizers/llama', use_fast=False, trust_remote_code=True)
    # print(tokenizer)

    config = XModelConfig.from_name(model_name)
    model = XModelForCausalLM(config)
    PATH = 'working_dir_%s/pytorch_model.bin'%model_name
    model.load_state_dict(torch.load(PATH))
    model.eval()
    model.to(args.device)
    print('model loaded')

    acc = eval_winograd(model, tokenizer, args.device)
    print('acc: %.4f'%acc)