import argparse
import json
import os
import os.path
import random
import warnings

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from modeling_xmodel import XModelForCausalLM
from configuration_xmodel import XModelConfig

choices = ["A", "B", "C", "D"]


def format_example(example, include_answer=True, k=4):
    question = example['question'].strip()
    answer = example['answer'].strip()

    options = example['options']
    options = [op.strip() for op in options]

    prompt = f'Question: {question}\n'

    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], options[j])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += "{}\n\n".format(answer)
    return prompt


def gen_prompt(eval_version):
    dev_file = f'data/xdata_thai_dev_{eval_version}.json'
    with open(dev_file) as fp:
        lines = fp.readlines()

    dev_df = [json.loads(line) for line in lines]
    dev_df = dev_df[:num_fewshot]

    prompt = "The following are multiple choice questions (with answers) about Thai language knowledge.\n\n"

    for example in dev_df:
        prompt += format_example(example)
    return prompt


def extract_answer(text):
    if text[:1] in choices:
        return text[:1]
    elif '\n' in text:
        return text.split('\n')[0]
    else:
        return text


@torch.no_grad()
def eval(model, tokenizer, eval_version):
    test_file = f'data/xdata_thai_test_{eval_version}.json'
    with open(test_file) as fp:
        lines = fp.readlines()

    test_df = [json.loads(line) for line in lines]

    cors = []
    labels = []
    preds = []

    for example in tqdm(test_df):
        # get prompt and make sure it fits
        prompt_end = format_example(example, include_answer=False)
        train_prompt = gen_prompt(eval_version)
        prompt = train_prompt + prompt_end
        prompt = prompt.replace('</s>', '')
        if args.apply_chat_template:
            messages = [{'role': 'user', 'content': prompt}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            # print(prompt)
        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True,
                                  max_length=args.context_size-100).to(model.device)
        num_tokens = inputs.shape[-1]
        # print('inputs: ' + str(inputs))
        # print('inputs.size(): ' + str(inputs.size()))
        # print('inputs.shape: ' + str(inputs.shape))

        # print('=================================================')
        # print('prompt: ' + str(prompt))

        label = example['answer']

        outputs = model.generate(
            inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(outputs[0][num_tokens:]).strip()
        # print('text: ' + str(text))

        pred = extract_answer(text)

        cor = pred == label
        cors.append(cor)
        labels.append(label)
        preds.append(pred)

    # print('preds: ' + str(sorted(preds)))
    # print('labels: ' + str(sorted(labels)))

    acc = np.mean(cors)
    print(args.model_path)
    print("Average accuracy {:.3f}".format(acc))


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, skip_special_tokens=True, add_bos_token=False,
                                              add_eos_token=False, clean_up_tokenization_spaces=True,
                                              use_fast=False,
                                              trust_remote_code=True)
    ckp_path = f'{args.model_path}/pytorch_model_fsdp.bin'
    # if not os.path.isfile(ckp_path):
    #     ckp_path = f'{args.model_path}/pytorch_model.bin'

    # print(f"Loading model from {args.model_path}")
    if not os.path.isfile(ckp_path):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, ignore_mismatched_sizes=True)
    else:
        config = XModelConfig.from_name('xl')
        model = XModelForCausalLM(config)
        model.load_state_dict(torch.load(ckp_path, map_location=f'cuda'))

    model.eval()
    model.cuda()
    # print(model)
    # print(tokenizer)
    eval(model, tokenizer, args.eval_version)


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    torch.manual_seed(42)
    # To avoid warnings about parallelism in tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_fewshot", "-k", type=int, default=3)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model_path", "-m", type=str,
                        default="")
    parser.add_argument("--context_size", type=int, default=2048)
    parser.add_argument("--eval_version", "-v", type=str, default="v2",
                        help="Evaluation version for different test and dev datasets")
    parser.add_argument("--apply_chat_template", "-t", type=bool, default=False,
                        help="apply_chat_template,default is chatml")
    args = parser.parse_args()

    num_fewshot = args.num_fewshot

    # print(args)

    main(args)
