import argparse
import nltk
import json
import sacrebleu
from tqdm import tqdm
import pyarrow.parquet as pq

import sys
from pathlib import Path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from models.modeling_xmodel import XModelForCausalLM
from models.configuration_xmodel import XModelConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer



data_path = '/data/wangyichuan/llm_pretrain_xmodel/data/sumation/sum_of_digits.jsonl'



def get_res(text):
    i = 0
    for i in range(len(text)-1):
        if text[i] == '\n':
            return text[:i]
        if text[i] == '.' and text[i+1] in [' ', '\n']:
            return text[:i+1]
    return text



def eval_sumation(model, tokenizer, device, data_path=data_path):
    with open(data_path, 'r') as f:
        acc = 0
        prompt = 'Calculate the following addition equation.'
        for i in range(5):
            line = json.loads(f.readline())
            prompt += line['context'] + line['completion']
        for i in tqdm(range(5, 150)):
            line = json.loads(f.readline())
            input_text, ans = prompt + line['context'], line['completion']
            inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
            num_tokens = inputs.shape[-1]
            outputs = model.generate(inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(outputs[0][num_tokens:]).strip()
            res = get_res(text)
            print(res, ans)
            try:
                res, ans = int(res), int(ans)
                acc += 1*(res == ans)
            except:
                continue
        return acc / 145

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:7", help="")
    args = parser.parse_args()

    # model_dir = 'working_dir'
    # model_name = 'medium'
    # tokenizer = transformers.AutoTokenizer.from_pretrained('/data/wangyichuan/llm_pretrain_xmodel/tokenizers/llama', use_fast=False, trust_remote_code=True)
    # config = XModelConfig.from_name(model_name)
    # model = XModelForCausalLM(config)
    # PATH = 'working_dir_%s/pytorch_model.bin'%model_name
    # model.load_state_dict(torch.load(PATH))
    # model.eval()
    # model.to(args.device)
    # print('model loaded')

    # bleu = eval_wmt(model, tokenizer, args.device)
    # print('bleu: %.4f'%bleu)


    # GPT-2
    model_path = "/data/wangyichuan/llm_pretrain_xmodel/gpt/gpt2-large"

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(args.device)
    print('model loaded')

    acc = eval_sumation(model, tokenizer, args.device)
    print('acc: %.4f'%acc)