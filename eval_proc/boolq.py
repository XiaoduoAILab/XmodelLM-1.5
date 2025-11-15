import re
from collections import Counter
import argparse
import json
import torch
import transformers
import nltk
import sacrebleu
import pandas as pd
from tqdm import tqdm
import pyarrow.parquet as pq

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import sys
from pathlib import Path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from models.modeling_xmodel import XModelForCausalLM
from models.configuration_xmodel import XModelConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer

import numpy as np
import random

def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



data = pd.read_csv('/data/wangyichuan/llm_pretrain_xmodel/data/boolq/boolq_train.csv')
shots = 2
n = 200

index, temp = [], True
for i in range(len(data)):
    if data['answer'][i] == temp:
        index.append(i)
        temp = not temp


def get_res(text):
    i = 0
    for i in range(len(text)-1):
        if text[i] in [',', '.', '<', '\n']:
            text = text[:i]
            break
    if text.replace(' ','').lower() in ['true', 'yes']:
        return True
    elif text.replace(' ','').lower() in ['false', 'no']:
        return False
    return text


def eval_boolq(model, tokenizer, device, data=data):
    true, pred = [], []
    prompt = ''
    for i in range(shots):
        id = index[i]
        prompt += 'passage:\n'+data['passage'][id]+'\n'+'Question:\n'+data['question'][id]+'\n'+'Answer:\n'+str(data['answer'][id])+'\n\n'

    for i in tqdm(range(shots, shots+n)):
        id = index[i]
        input_text, ans = prompt+'passage:\n'+data['passage'][id]+'\n'+'Question:\n'+data['question'][id]+'\n'+'Answer:\n', data['answer'][id]
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
        num_tokens = inputs.shape[-1]
        print(model.device, inputs.device)
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(outputs[0][num_tokens:]).strip()
        res = get_res(text)
        
        true.append(1*ans)
        if not isinstance(res, str):
            pred.append(1*res)
        else:
            print(res)
            pred.append(-1)
    
    print(confusion_matrix(true, pred))

    return accuracy_score(true, pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:6", help="")
    args = parser.parse_args()

    setup_seeds()

    # GPT-2
    '''
    small: 0.52
    medium: 0.54
    large: 0.41
    xl: 0.53
    '''

    model_path = "/data/wangyichuan/llm_pretrain_xmodel/gpt/gpt2"

    # tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    # model = GPT2LMHeadModel.from_pretrained(model_path).to(args.device)
    # print('model loaded')

    # Xmodel
    '''
    small: 0.49
    medium: 0.48
    '''

    model_name = 'medium'
    PATH = '/data4/liuyang/checkpoints/c_line/medium/iter-350000'

    # tokenizer = transformers.AutoTokenizer.from_pretrained('/data/wangyichuan/llm_pretrain_xmodel/tokenizers/llama', use_fast=False, trust_remote_code=True)
    # config = XModelConfig.from_name(model_name)
    # model = XModelForCausalLM(config)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()
    # model.to(args.device)
    # print('model loaded')

    # Open-src models
    '''
    Qwen-1.8B: 0.625
    Qwen-7B: 0.83

    phi1: 
    '''
    path = '/data1/liuyang/models/Qwen/Qwen-7B'
    tokenizer = transformers.AutoTokenizer.from_pretrained(path, add_bos_token=False, add_eos_token=False, clean_up_tokenization_spaces=True, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(path, device_map='auto', trust_remote_code=True).to(args.device)
    print('model loaded')



    acc = eval_boolq(model, tokenizer, args.device)
    print(model_path)
    print('acc: %.4f'%acc)