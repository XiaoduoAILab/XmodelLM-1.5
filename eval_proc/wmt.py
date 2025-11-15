import argparse
import nltk
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


pq_path = str(wd)+'/data/wmt14/dev.parquet'
text = pq.read_table(pq_path).to_pandas()['translation']
prompt = ''
seed = [3, 35, 156, 356, 2549]
for s in seed:
    prompt += '%s => %s\n\n'%(text[s]['fr'], text[s]['en'])


pq_path = str(wd)+'/data/wmt14/test.parquet'
text = pq.read_table(pq_path).to_pandas()['translation'][:300]


def get_res(text):
    i = 0
    for i in range(len(text)-1):
        if text[i] == '\n':
            return text[:i]
        if text[i] == '.' and text[i+1] in [' ', '\n']:
            return text[:i+1]
    return text


def cal_bleu(res, ans):
    candidate_tokens = nltk.word_tokenize(res)
    reference_tokens = nltk.word_tokenize(ans)
    return nltk.translate.bleu_score.sentence_bleu([reference_tokens], candidate_tokens)


def eval_wmt(model, tokenizer, device, text=text):
    bleu = []
    for t in tqdm(text):
        input_text, ans = prompt + '%s => '%t['fr'], t['en']
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
        num_tokens = inputs.shape[-1]
        outputs = model.generate(inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(outputs[0][num_tokens:]).strip()
        res = get_res(text)
        try:
            score = sacrebleu.raw_corpus_bleu(res, [ans]).score
        except:
            score = 0
        bleu.append(score)
    return sum(bleu) / len(bleu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:6", help="")
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
    model_path = "/data/wangyichuan/llm_pretrain_xmodel/gpt/gpt2-xl"

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(args.device)
    print('model loaded')

    bleu = eval_wmt(model, tokenizer, args.device)
    print('bleu: %.4f'%bleu)