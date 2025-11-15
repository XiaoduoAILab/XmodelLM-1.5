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
import pandas as pd
import torch
import transformers
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))
from utils import load_jsonl
from models.modeling_xmodel import XModelForCausalLM
from models.configuration_xmodel import XModelConfig
from categories import subcategories, categories

choices = ["A", "B", "C", "D"]


def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_dir", type=str, default="/data1/liuyang/llm_pretrain_xmodel/out_redpajama/medium/",
                        help="folder of checkpoints")

    parser.add_argument("--data_dir", type=str, default='/data1/liuyang/llm_evals/llm_mmlu/', help="data path")

    return parser


def format_subject(subject):
    return subject.replace('_', ' ')


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:  "
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    all_labels = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        while inputs.shape[-1] > context_size:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        label = test_df.iloc[i, test_df.shape[1] - 1]

        outputs = model.generate(inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True,
                                 pad_token_id=tokenizer.eos_token_id)

        logits = outputs.scores[0][0]

        logits = torch.nn.functional.softmax(logits, dim=0, dtype=torch.float32)
        probs = torch.tensor(
            [
                logits[tokenizer("A").input_ids[0]],
                logits[tokenizer("B").input_ids[0]],
                logits[tokenizer("C").input_ids[0]],
                logits[tokenizer("D").input_ids[0]],
            ]
        )

        probs = probs.detach().cpu().numpy()
        probs = np.array([probs[0], probs[1], probs[2], probs[3], (1 - sum(probs[:4]))])

        probs = (probs)

        pred = {0: "A", 1: "B", 2: "C", 3: "D", 4: "Z"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)
        all_labels.append(label)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    # print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs, all_labels


def eval_mmlu(model, tokenizer):
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "data/test"))
            if "_test.csv" in f
        ]
    )

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}
    all_probs = []
    all_labels = []

    for subject in tqdm(subjects):
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "data/dev", subject + "_dev.csv"), header=None
        )[: ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "data/test", subject + "_test.csv"), header=None
        )

        cors, acc, probs, labels = eval(subject, model, tokenizer, dev_df, test_df)
        all_probs.append(probs.tolist())
        all_labels.append(labels)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

    weighted_acc = np.mean(np.concatenate(all_cors))
    # print("Average accuracy: {:.3f}".format(weighted_acc))
    return weighted_acc


if __name__ == "__main__":
    torch.manual_seed(42)

    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    ckpt_dir = args.ckpt_dir
    data_dir = args.data_dir

    ntrain = 5
    context_size = 2048

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

    filename = f'trends/mmlu/result/mmlu_{model_name}.jsonl'
    if os.path.isfile(filename):
        result = load_jsonl(filename)
        done_iters = [item['num_iter'] for item in result]
        tasks = [t for t in tasks if t['num_iter'] not in done_iters]

    tasks.sort(key=lambda t: t['num_iter'])
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

            accuracy = eval_mmlu(model, tokenizer)

            json_obj = {'num_iter': num_iter, 'accuracy': accuracy}
            json_str = json.dumps(json_obj)

            fp.write(json_str)
            fp.write('\n')
            fp.flush()
