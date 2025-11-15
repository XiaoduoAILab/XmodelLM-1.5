import os
import json
import argparse
from tqdm import tqdm
from mteb import MTEB

import sys
from pathlib import Path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from models.modeling_xmodel import XModelForCausalLM
from models.configuration_xmodel import XModelConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from utils import extract_embeddings


class MyModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def encode(self, sentences, batch_size=8, **kwargs):
        return extract_embeddings(self.model, self.tokenizer, sentences, batch_size)


def eval_mteb_cls(model, tokenizer, task="Banking77Classification"):
    tokenizer.pad_token = tokenizer.eos_token
    model = MyModel(model, tokenizer)
    evaluation = MTEB(tasks=[task])
    evaluation.run(model, output_folder="results")
    
    js_file_path = 'results/%s.json'%task
    with open(js_file_path, 'r') as f:
        js = json.load(f)
        score = js['test']['f1']

    os.remove(js_file_path)
    
    return score



def eval_mteb_sts(model, tokenizer, task="SICK-R"):
    tokenizer.pad_token = tokenizer.eos_token
    model = MyModel(model, tokenizer)
    evaluation = MTEB(tasks=[task])
    evaluation.run(model, output_folder="results")
    
    js_file_path = 'results/%s.json'%task
    
    with open(js_file_path, 'r') as f:
        js = json.load(f)
        score = js['test']['cos_sim']['pearson'] + js['test']['cos_sim']['spearman'] + js['test']['euclidean']['pearson'] + js['test']['euclidean']['spearman'] + js['test']['manhattan']['pearson'] + js['test']['manhattan']['spearman']
        score = score / 6

    os.remove(js_file_path)
    
    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:6", help="")
    args = parser.parse_args()

    model_path = "/data/wangyichuan/llm_pretrain_xmodel/gpt/gpt2"

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(args.device)
    print('model loaded')

    tokenizer.pad_token = tokenizer.eos_token
    model = MyModel(model, tokenizer)
    sentence = ['hello world!', 'shut the fuck up.', '']
    print(model.encode(sentence))

    # print(eval_mteb(model, tokenizer))

    # tokenizer.pad_token = tokenizer.eos_token
    # model = MyModel(model, tokenizer)
    # evaluation = MTEB(tasks=['SICK-R'])
    # evaluation.run(model, output_folder="results")