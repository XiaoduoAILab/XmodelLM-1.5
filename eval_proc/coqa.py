import re
from collections import Counter
import argparse
import json
import torch
import transformers
import nltk
import sacrebleu
from tqdm import tqdm
import pyarrow.parquet as pq

from utils import eval_ppl_sent

import sys
from pathlib import Path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from models.modeling_xmodel import XModelForCausalLM
from models.configuration_xmodel import XModelConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer


data = json.load(open(str(wd)+'/data/coqa/coqa.json', 'r'))
data = data['data'][:30]


def get_res(text):
    i = 0
    for i in range(len(text)-1):
        if text[i] == '\n':
            return text[:i]
        if text[i] == '.' and text[i+1] in [' ', '\n']:
            return text[:i+1]
    return text

def tokenize(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return Counter(words)

def f1_score(prediction, reference):
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    common = pred_tokens & ref_tokens
    common_count = sum(common.values())
    if common_count == 0:
        return 0.0

    precision = common_count / sum(pred_tokens.values())
    recall = common_count / sum(ref_tokens.values())
    f1 = (2 * precision * recall) / (precision + recall)

    return f1

def eval_coqa(model, tokenizer, device, data=data):
    f1 = []
    for item in tqdm(data):
        if len(item['questions']) <= 5:
            continue
        prompt = 'story:\n\n'+item['story']+'\n\n'+'Question:\n'+item['questions'][0]['input_text']+'\n\n'+'Answer:\n'+item['answers'][0]['input_text']+'\n\n'
        prompt += 'Question:\n'+item['questions'][1]['input_text']+'\n\n'+'Answer:\n'+item['answers'][1]['input_text']+'\n\n'


        for i in range(2, len(item['questions'])):
            input_text = prompt+'Question:\n'+item['questions'][i]['input_text']+'\n\n'+'Answer:\n'
            ans = item['answers'][i]['input_text']
            prompt += 'Question:\n'+item['questions'][i]['input_text']+'\n\n'+'Answer:\n'+item['answers'][i]['input_text']+'\n\n'
            inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
            num_tokens = inputs.shape[-1]
            outputs = model.generate(inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(outputs[0][num_tokens:]).strip()
            res = get_res(text)
            score = f1_score(res, ans)
            f1.append(score)
    return sum(f1) / len(f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:7", help="")
    args = parser.parse_args()

    # GPT-2
    model_path = "/data/wangyichuan/llm_pretrain_xmodel/gpt/gpt2"

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(args.device)
    print('model loaded')

    f1 = eval_coqa(model, tokenizer, args.device)
    print(model_path)
    print('f1: %.4f'%f1)