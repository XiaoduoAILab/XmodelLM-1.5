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
import re
import sys
from pathlib import Path

import torch
import transformers
from tqdm import tqdm
from transformers import StoppingCriteriaList

# support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))
from data import HUMAN_EVAL
from data import write_jsonl, read_problems
from evaluation import evaluate_functional_correctness
from models.modeling_xmodel import XModelForCausalLM
from models.configuration_xmodel import XModelConfig
from utils import load_jsonl


def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_dir", type=str, default="/data1/liuyang/llm_pretrain_xmodel/out_redpajama/medium/",
                        help="folder of checkpoints")

    parser.add_argument("--data_dir", type=str, default='/data1/liuyang/llm_evals/llm_gsm8k/', help="data path")

    return parser


def clean_code(text):
    if '\n\n\n' in text:
        text = text.split('\n\n\n')[0]
    else:
        pattern = '[\n][\n][\S]'
        result = re.search(pattern, text)
        if result:
            position = result.span()[0]
            text = text[:position]

    return text


def custom_stopping_criteria(input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
    # last_token = input_ids[0][-1]
    # for stop in stop_words_ids:
    #     if tokenizer.decode(stop) == tokenizer.decode(last_token):
    #         return True
    if input_ids[0][-1] == 13 and input_ids[0][-2] == 13 and input_ids[0][-3] == 13:
        return True

    return False


@torch.no_grad()
def generate_one_completion(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_new_tokens=seq_len, pad_token_id=tokenizer.eos_token_id,
                             stopping_criteria=StoppingCriteriaList([custom_stopping_criteria]))
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    text = text[len(prompt):]
    text = clean_code(text)

    return text


def eval_human_eval(model, tokenizer):
    problems = read_problems()
    sample_file = f'human_eval_{model_name}.txt'

    # num_samples_per_task = 200
    num_samples_per_task = 1
    samples = []

    for task_id in tqdm(problems):
        for _ in range(num_samples_per_task):
            completion = generate_one_completion(problems[task_id]["prompt"])
            samples.append(dict(task_id=task_id, completion=completion))

    write_jsonl(sample_file, samples)

    k = "1,10,100"
    n_workers = 1
    timeout = 3.0
    problem_file = HUMAN_EVAL

    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    return results['pass@1']


if __name__ == "__main__":
    torch.manual_seed(42)

    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    ckpt_dir = args.ckpt_dir

    seq_len = 256
    num_beams = 1
    do_sample = False

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

    filename = f'trends/human_eval/result/human_eval_{model_name}.jsonl'
    if os.path.isfile(filename):
        result = load_jsonl(filename)
        done_iters = [item['num_iter'] for item in result]
        tasks = [t for t in tasks if t['num_iter'] not in done_iters]
    print('len(tasks): ' + str(len(tasks)))
    with open(filename, 'a+') as fp:

        for task in tqdm(tasks):
            ckpt_path = task['ckpt_path']
            ckpt_name = task['ckpt_name']
            num_iter = task['num_iter']

            model_path = '%s/pytorch_model.bin' % ckpt_path
            model.load_state_dict(torch.load(model_path))
            model.to('cuda')
            model.eval()

            accuracy = eval_human_eval(model, tokenizer)

            json_obj = {'num_iter': num_iter, 'accuracy': accuracy}
            json_str = json.dumps(json_obj)

            fp.write(json_str)
            fp.write('\n')
            fp.flush()
