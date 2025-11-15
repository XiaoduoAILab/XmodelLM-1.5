import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from eval_proc.utils import deploy_model

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def do_task(tasks):
    start = time.time()
    command = f"lm_eval --model hf --model_args pretrained=working_dir_{eval_name},trust_remote_code=True --tasks {str_tasks} --device {device} --batch_size {args.batch_size}"
    print('command: ' + str(command))

    result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    print('stdout: ' + str(result.stdout))

    matches = re.findall(f"\|{args.metric}\s*\|↑\s*\|\s*([\d.]+)\s*\|", result.stdout, re.DOTALL)
    print('matches: ' + str(matches))

    for i, task in enumerate(tasks):
        with open(f'{output_folder}/{task}.jsonl', 'a+') as fp:
            match = matches[i]
            acc = float(match)
            line = json.dumps(dict(ckp_name=ckp_name, acc=acc))
            print(line)
            fp.write(line + '\n')

    print('time: ', time.time() - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=str, default="/data4/liuyang/checkpoints/xl_f_line/")
    parser.add_argument("--line", type=str, default="f_line")
    parser.add_argument("--model_name", type=str, default="xl")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizers/xmodel/v11/xmodel_65280")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--tasks", type=str, default='wmt14-en-fr,wmt14-fr-en,xcopa_th,xcopa_zh')
    parser.add_argument("--metric", type=str, default='acc')
    args = parser.parse_args()

    device = args.device
    line = args.line
    model_name = args.model_name
    # model_path = os.path.join(args.checkpoints_dir, args.model_name)
    model_path = args.checkpoints_dir
    print('model_name: ' + str(model_name))

    output_folder = f'eval_result/{line}/{model_name}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # load data
    # datasets = ['arc_challenge', 'arc_easy', 'boolq', 'openbookqa', 'piqa', 'winogrande']

    # if model_name in ['nano', 'micro']:
    #     datasets = ['arc_easy', 'hellaswag', 'piqa']
    # elif model_name in ['tiny', 'small', 'medium']:
    # datasets = ['arc_challenge', 'arc_easy', 'boolq', 'hellaswag', 'openbookqa', 'piqa', 'sciq', 'winogrande']
    # datasets = ['arc_ar', 'arc_fr', 'arc_fr', 'm_mmlu_ar', 'm_mmlu_fr', 'm_mmlu_en', 'm_mmlu_zh']
    str_tasks = args.tasks
    tasks = str_tasks.split(',')
    tasks = sorted(tasks)
    print('tasks:' + str(tasks))
    # load tokenizer

    while True:

        dirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
        dirs = sorted(dirs)
        # dirs = [dirs[i*5] for i in range(len(dirs)//5)]

        eval_name = tasks[0]
        filename = f'{output_folder}/{eval_name}.jsonl'
        if os.path.isfile(filename):
            with open(filename, 'r') as fp:
                lines = fp.readlines()

            ckps = []
            for line in lines:
                item = json.loads(line)
                ckps.append(item['ckp_name'])

            dirs = [d for d in dirs if d not in ckps]

        print(len(dirs))
        if len(dirs) == 0:
            break

        for ckp_name in tqdm(dirs):
            ckp_folder = os.path.join(model_path, ckp_name)

            try:
                deploy_model(ckp_folder, eval_name, tokenizer_path=args.tokenizer_path)

                print(f'{args.tasks} <- {ckp_name}')
                do_task(tasks)

            except subprocess.CalledProcessError as error:
                print(error)
                time.sleep(5)
