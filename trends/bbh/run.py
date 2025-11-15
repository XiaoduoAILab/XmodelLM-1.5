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

import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import StoppingCriteriaList

# support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))
from utils import load_jsonl
from models.modeling_xmodel import XModelForCausalLM
from models.configuration_xmodel import XModelConfig


def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_dir", type=str, default="/data1/liuyang/llm_pretrain_xmodel/out_redpajama/medium/",
                        help="folder of checkpoints")

    parser.add_argument("--data_dir", type=str, default='/data1/liuyang/llm_evals/llm_bbh/', help="data path")

    return parser


MULTIPLE_CHOICE_TASKS = [
    'temporal_sequences', 'disambiguation_qa', 'date_understanding', 'tracking_shuffled_objects_three_objects',
    'penguins_in_a_table',
    'geometric_shapes', 'snarks', 'ruin_names', 'tracking_shuffled_objects_seven_objects',
    'tracking_shuffled_objects_five_objects',
    'logical_deduction_three_objects', 'hyperbaton', 'logical_deduction_five_objects',
    'logical_deduction_seven_objects', 'movie_recommendation',
    'salient_translation_error_detection', 'reasoning_about_colored_objects',
]
FREE_FORM_TASKS = [
    'multistep_arithmetic_two', 'navigate', 'dyck_languages', 'word_sorting', 'sports_understanding',
    'boolean_expressions', 'object_counting', 'formal_fallacies', 'causal_judgement', 'web_of_lies',
]


def custom_stopping_criteria(input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
    # last_token = input_ids[0][-1]
    # for stop in stop_words_ids:
    #     if tokenizer.decode(stop) == tokenizer.decode(last_token):
    #         return True
    if input_ids[0][-1] == 13 and input_ids[0][-2] == 13:
        return True

    return False


@torch.no_grad()
def generate_one_completion(prompt, model, tokenizer):
    try:
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        num_tokens = inputs.shape[-1]
        outputs = model.generate(inputs, max_new_tokens=seq_len, pad_token_id=tokenizer.eos_token_id,
                                 stopping_criteria=StoppingCriteriaList([custom_stopping_criteria]))
        text = tokenizer.decode(outputs[0][num_tokens:]).strip()
    except RuntimeError as err:
        print(err)
        text = ""
    return text


def extract_ans(ans, mode):
    ans_line = ans.split('answer is ')
    # Expect to see 'answer is'. If not return ""
    if len(ans_line) == 1:
        return ans
    else:
        ans = ans_line[-1].strip()

    if mode == 'multiple_choice':
        options = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)', '(M)', '(N)',
                   '(O)', '(P)', '(Q)', '(R)', '(S)', '(T)', '(U)', '(V)', '(W)', '(X)', '(Y)', '(Z)']
        # for option in options:
        #     if option in ans:
        #         ans = option[1]
        #         break
        # return ans

        positions = []
        for opt in options:
            if opt in ans:
                positions.append(ans.index(opt))
            else:
                positions.append(float('inf'))

        if positions:
            return options[np.argmin(positions)][1]
        else:
            return 'Z'

    elif mode == 'free_form':
        if '.' in ans:
            pos = ans.index('.')
            ans = ans[:pos]

        return ans


def run_tasks(args, tasks, model, tokenizer, mode):
    logname = model_name + '.txt'
    if os.path.exists(logname):
        os.remove(logname)

    acc_total = 0
    num_total = 0

    with open(logname, 'w') as fd:
        for task in tqdm(tasks):
            # print('Testing %s ...' % task)
            fd.write('Testing {} ...'.format(task))

            acc = 0
            task_data = json.load(open(f'{data_dir}/%s.json' % task))
            task_prompt = open(f'{prompts_dir}/%s.txt' % task, 'r').read()
            task_prompt = task_prompt[111:]

            num_examples = len(task_data['examples'])
            for i in range(num_examples):
                q_ = task_data['examples'][i]
                q = '\n\nQ: ' + q_['input']

                prompt_q = task_prompt + q + "\nA: Let's think step by step."
                prompt = "Follow the given examples and answer the question.\n\n" + prompt_q

                response = generate_one_completion(prompt, model, tokenizer)
                ans_model = response

                ans_ = extract_ans(response, mode)

                if mode == 'multiple_choice':
                    a = q_['target'][1]
                elif mode == 'free_form':
                    a = q_['target']

                fd.write('%s\nA_model:\n%s\nA_target:\n%s\n\n' % (q, ans_model, a))

                correct = ans_ == a
                if correct:
                    acc += 1
                    acc_total += 1
                num_total += 1

                log_line = json.dumps(dict(question=q_['input'], answer=a, output=ans_, correct=correct))

            # print('%s acc %.4f' % (task, acc / num_examples))

    return acc_total, num_total


def eval_bbh(model, tokenizer, multiple_choice_tasks=MULTIPLE_CHOICE_TASKS, free_form_tasks=FREE_FORM_TASKS):
    acc_total = 0
    num_total = 0

    # run_multiple_choice = task == 'all' or task == 'multiple_choice'
    # run_free_form = task == 'all' or task == 'free_form'

    # if run_multiple_choice:
    acc, num = run_tasks(args, multiple_choice_tasks, model, tokenizer, mode='multiple_choice')
    acc_total += acc
    num_total += num
    # print('%s acc %.4f' % ('multiple_choice_tasks', acc / num))

    # if run_free_form:
    acc, num = run_tasks(args, free_form_tasks, model, tokenizer, mode='free_form')
    acc_total += acc
    num_total += num
    # print('%s acc %.4f' % ('free_form_tasks', acc / num))

    # print('%s acc %.4f' % ('total', acc_total / num_total))
    accuracy = acc_total / num_total
    return accuracy


if __name__ == "__main__":
    torch.manual_seed(42)

    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    ckpt_dir = args.ckpt_dir

    seq_len = 256
    num_beams = 1
    do_sample = False
    task = 'all'
    data_dir = os.path.join(args.data_dir, 'bbh')
    prompts_dir = os.path.join(args.data_dir, 'cot-prompts')

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
        if num_iter <= 10000 or num_iter % 10000 == 0:
            tasks.append(dict(ckpt_path=ckpt_path, ckpt_name=ckpt_name, num_iter=num_iter))

    filename = f'trends/bbh/result/bbh_{model_name}.jsonl'
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

            accuracy = eval_bbh(model, tokenizer)

            json_obj = {'num_iter': num_iter, 'accuracy': accuracy}
            json_str = json.dumps(json_obj)

            fp.write(json_str)
            fp.write('\n')
            fp.flush()
