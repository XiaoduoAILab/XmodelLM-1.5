# Copyright 2024 XiaoDuo AI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import transformers
from tqdm import tqdm
from transformers import StoppingCriteriaList

# support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))
from utils import load_jsonl
from dataset import get_examples, extract_answer, INVALID_ANS
from models.modeling_xmodel import XModelForCausalLM
from models.configuration_xmodel import XModelConfig


def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_dir", type=str, default="/data1/liuyang/llm_pretrain_xmodel/out_redpajama/medium/",
                        help="folder of checkpoints")

    parser.add_argument("--data_dir", type=str, default='/data1/liuyang/llm_evals/llm_gsm8k/', help="data path")

    return parser


def custom_stopping_criteria(input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
    # last_token = input_ids[0][-1]
    # for stop in stop_words_ids:
    #     if tokenizer.decode(stop) == tokenizer.decode(last_token):
    #         return True
    if input_ids[0][-1] == 13 and input_ids[0][-2] == 13:
        return True

    return False


@torch.no_grad()
def generate(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    num_tokens = inputs.shape[-1]
    outputs = model.generate(inputs, max_new_tokens=seq_len, pad_token_id=tokenizer.eos_token_id,
                             stopping_criteria=StoppingCriteriaList([custom_stopping_criteria]))
    text = tokenizer.decode(outputs[0][num_tokens:]).strip()
    return text


def eval_gsm8k(model, tokenizer):
    splitter = '\n\n'
    num_correct = 0
    num_processed = 0

    test_examples = get_examples("test", data_dir=args.data_dir)
    num_examples = len(test_examples)
    prompt = open(prompt_file, 'r').read()

    for i in tqdm(range(num_examples)):
        example = test_examples[i]
        qn = example["question"].strip()
        # an = example["answer"].strip()
        input_text = '{}\nQuestion: {}\nLet\'s think step by step\n'.format(prompt, qn)
        # print("==========================================")
        # print(input_text)

        output_text = generate(input_text)
        if splitter in output_text:
            output_text = output_text.split(splitter)[0]
        model_output = extract_answer(output_text.replace('$', ''))
        gt_answer = extract_answer(example["answer"])

        # print('Answer: {}'.format(gt_answer))
        # print('Model output: {}'.format(model_output))

        num_processed += 1

        correct = False
        if model_output != INVALID_ANS:
            try:
                correct = eval(model_output) == eval(gt_answer)
            except Exception as ex:
                print(ex)

        if correct:
            num_correct += 1

    accuracy = num_correct * 100 / num_examples
    return accuracy


if __name__ == "__main__":
    torch.manual_seed(42)

    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    ckpt_dir = args.ckpt_dir
    data_dir = args.data_dir

    seq_len = 256
    num_beams = 1
    do_sample = False

    prompt_file = os.path.join(args.data_dir, 'lib_prompt/prompt_original_5_shot.txt')

    tokenizer = transformers.AutoTokenizer.from_pretrained('tokenizers/llama', skip_special_tokens=True,
                                                           add_bos_token=False,
                                                           add_eos_token=False,
                                                           clean_up_tokenization_spaces=True,
                                                           use_fast=False,
                                                           trust_remote_code=True)
    print(tokenizer)

    if ckpt_dir.endswith('/'):
        ckpt_dir = ckpt_dir[:-1]

    model_name = ckpt_dir.split('/')[-1]
    print('model_name: ' + str(model_name))

    config = XModelConfig.from_name(model_name)
    model = XModelForCausalLM(config)

    tasks = []
    for ckpt_name in [f for f in os.listdir(ckpt_dir) if f.startswith('iter-')]:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        num_iter = int(ckpt_name[5:])
        # if num_iter % 10000 == 0:
        tasks.append(dict(ckpt_path=ckpt_path, ckpt_name=ckpt_name, num_iter=num_iter))

    filename = f'trends/gsm8k/result/gsm8k_{model_name}.jsonl'
    if os.path.isfile(filename):
        result = load_jsonl(filename)
        done_iters = [item['num_iter'] for item in result]
        tasks = [t for t in tasks if t['num_iter'] not in done_iters]

    print('len(tasks): ' + str(len(tasks)))
    # print('tasks: ' + str(tasks))

    with open(filename, 'a+') as fp:

        for task in tqdm(tasks):
            ckpt_path = task['ckpt_path']
            ckpt_name = task['ckpt_name']
            num_iter = task['num_iter']

            model_path = '%s/pytorch_model.bin' % ckpt_path
            model.load_state_dict(torch.load(model_path))
            model.to('cuda')
            model.eval()

            accuracy = eval_gsm8k(model, tokenizer)

            json_obj = {'num_iter': num_iter, 'accuracy': accuracy}
            json_str = json.dumps(json_obj)

            fp.write(json_str)
            fp.write('\n')
            fp.flush()
