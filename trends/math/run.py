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

# support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))
from utils import load_jsonl
from math_equivalence import is_equiv
from models.modeling_xmodel import XModelForCausalLM
from models.configuration_xmodel import XModelConfig


def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_dir", type=str, default="/data1/liuyang/llm_pretrain_xmodel/out_redpajama/medium/",
                        help="folder of checkpoints")

    parser.add_argument("--data_dir", type=str, default='/data1/liuyang/llm_evals/llm_math/', help="data path")

    return parser


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


@torch.no_grad()
def generate_one_completion(prompt, model, tokenizer, num_tokens=20):
    try:
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        num_tokens = inputs.shape[-1]

        outputs = model.generate(inputs, max_new_tokens=num_tokens, pad_token_id=tokenizer.eos_token_id)

        text = tokenizer.decode(outputs[0][num_tokens:]).strip()
    except RuntimeError as err:
        print(err)
        text = ""

    return text


def call_engine(train_prompt, problem, model, tokenizer):
    '''
    Given a problem, returns the most likely answer determined by the engine
    '''
    test_question = "\n###\nProblem: " + problem + "\n" + "Answer: $"
    prompt = train_prompt + test_question

    output_str = generate_one_completion(prompt, model, tokenizer)
    output_full = output_str

    startindex = 0
    endindex = 1
    for token in output_full[startindex + 1:]:
        if token == "$" or token == "#" or token == "\n":
            break
        else:
            endindex += 1

    output = output_full[startindex:endindex]
    return output_full, output


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def eval_math(model, tokenizer):
    outputs = []
    answers = []
    types = []
    levels = []

    train_prompt = open(prompt_file, 'r').read()

    fnames_list = []
    finfo_list = []

    cors = {}
    subject_cors = {}
    level_cors = {}
    correct = 0
    total = 0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            fnames_list.append(os.path.join(subdir, file))
            finfo_list.append((subdir, file))

    num_examples = len(finfo_list)
    # print('num_examples: ' + str(num_examples))

    for i in tqdm(range(num_examples)):
        subdir, file = finfo_list[i]
        with open(os.path.join(subdir, file), 'r') as fp:
            try:
                problem_data = json.load(fp)
            except Exception as e:
                print(f"Error loading JSON from {file}", e)
                raise e
            prob_level = problem_data["level"]
            prob_type = problem_data["type"]
            try:
                prob_level = int(prob_level.split("Level ")[1])
            except:
                prob_level = None
            output_full, output = call_engine(train_prompt, problem_data["problem"], model, tokenizer)
            answer = remove_boxed(last_boxed_only_string(problem_data["solution"]))

            levels.append(prob_level)
            types.append(prob_type)
            outputs.append(output)
            answers.append(answer)

            try:
                equiv = is_equiv(output, answer)
            except:
                equiv = False
            if (prob_level, prob_type) in cors:
                cors[(prob_level, prob_type)].append(equiv)
            else:
                cors[(prob_level, prob_type)] = [equiv]
            if prob_level in level_cors:
                level_cors[prob_level].append(equiv)
            else:
                if prob_level is not None:
                    level_cors[prob_level] = [equiv]
            if prob_type in subject_cors:
                subject_cors[prob_type].append(equiv)
            else:
                if prob_type is not None:
                    subject_cors[prob_type] = [equiv]
            if equiv:
                correct += 1
            total += 1

    with open("outputs_answers.txt", "w+") as f:
        for k, (output, answer, prob_type, prob_level, fname) in enumerate(
                zip(outputs, answers, types, levels, fnames_list)):
            f.write("{} TYPE: {} | LEVEL: {} | OUTPUT: {} | ANSWER: {} | FNAME: {}\n".format(k, prob_type, prob_level,
                                                                                             output, answer, fname))

        # f.write("#####################\n")
        # also get accuracies for each
        for subject in ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry',
                        'Intermediate Algebra', 'Precalculus']:
            for level in range(1, 6):
                key = (level, subject)
                if key not in cors.keys():
                    print("Skipping", key)
                    continue
                cors_list = cors[key]
                # print("{} Level {} Accuracy = {}/{} = {:.3f}".format(subject, level, np.sum(cors_list), len(cors_list),
                #                                                      np.mean(cors_list)))
                # f.write(
                #     "{} Level {} Accuracy = {}/{} = {:.3f}\n".format(subject, level, np.sum(cors_list), len(cors_list),
                #                                                      np.mean(cors_list)))
        # print("#####################")
        f.write("#####################\n")
        for level in sorted(level_cors):
            if level not in level_cors.keys():
                print("Skipping", level)
                continue
            cors_list = level_cors[level]
            # print("Level {} Accuracy = {}/{} = {:.3f}".format(level, np.sum(cors_list), len(cors_list),
            #                                                   np.mean(cors_list)))
            # f.write("Level {} Accuracy = {}/{} = {:.3f}\n".format(level, np.sum(cors_list), len(cors_list),
            #                                                       np.mean(cors_list)))
        # print("#####################")
        # f.write("#####################\n")
        for subject in ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry',
                        'Intermediate Algebra', 'Precalculus']:
            if subject not in subject_cors.keys():
                print("Skipping", subject)
                continue
            cors_list = subject_cors[subject]
            # print("{} Accuracy = {}/{} = {:.3f}".format(subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
            # f.write(
            #     "{} Accuracy = {}/{} = {:.3f}\n".format(subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
        # print("#####################")
        # f.write("#####################\n")
        # print("Overall Accuracy = {}/{} = {:.3f}".format(correct, total, correct / total))
        # f.write("Overall Accuracy = {}/{} = {:.3f}\n".format(correct, total, correct / total))

    accuracy = correct / total
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
    rootdir = os.path.join(data_dir, "./MATH/test")

    prompt_file = os.path.join(data_dir, 'lib_prompt/prompt_ao_5_shot.txt')

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

    filename = f'trends/math/result/math_{model_name}.jsonl'
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

            accuracy = eval_math(model, tokenizer)

            json_obj = {'num_iter': num_iter, 'accuracy': accuracy}
            json_str = json.dumps(json_obj)

            fp.write(json_str)
            fp.write('\n')
            fp.flush()
