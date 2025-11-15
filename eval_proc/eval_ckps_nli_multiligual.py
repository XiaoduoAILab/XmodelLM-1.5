import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from glob import glob 

from tqdm import tqdm

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from utils import deploy_model

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def get_latest_result_file(ckp_name,args):
    # 获取文件夹中 result_*.json 的最新文件
    result_dir = f"/home/wangqun/{args.model_name}/eva_result/{ckp_name}_multiligual/" + f"{args.checkpoints_dir}{ckp_name}".replace("/", "__")
    # 判断 result_dir 是否存在
    if not os.path.exists(result_dir):
        # 如果不存在，返回 None 使得 if latest_file: 为 False
        return None
    result_files = os.listdir(result_dir)
    # print(result_files)
    # 使用正则表达式匹配文件名
    result_files = [f for f in result_files if re.match(r'results_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d{6}\.json', f)]
    if not result_files:
        raise FileNotFoundError(f"No result_*.json files found in {result_dir}")
    
    # 获取完整文件路径
    result_files = [os.path.join(result_dir, f) for f in result_files]
    
    # 按文件修改时间排序，获取最新的文件
    latest_file = max(result_files, key=os.path.getmtime)
    return latest_file

def get_acc_from_result_file(latest_file, task):
    # 读取最新的 result_*.json 文件并返回对应任务的 acc
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    # 提取指定任务的 acc
    if task in data['results']:
        task_results = data['results'][task]

        if 'acc,none' in task_results:
            return 'acc', task_results['acc,none']
        elif 'word_perplexity,none' in task_results:
            return 'word_perplexity', task_results['word_perplexity,none']
        else:
            raise ValueError(f"No acc or word_perplexity found for task {task} in {latest_file}")
    else:
        raise ValueError(f"No results found for task {task} in {latest_file}")
def do_task(tasks,ckp_name):
    start = time.time()
    command = f"CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com lm_eval --model hf --model_args pretrained={args.checkpoints_dir}{ckp_name},trust_remote_code=True --tasks {','.join(tasks)} --device {device} --batch_size {args.batch_size} --log_samples --output_path /home/wangqun/{args.model_name}/eva_result/{ckp_name}_multiligual 2>&1 | tee /home/wangqun/f_line/eva_result/{ckp_name}_multiligual.log"
    print(command)

    result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    print(result.stdout)

    # matches = re.findall(r"\|acc\s*\|\s*([\d.]+)\s*\|", result.stdout, re.DOTALL)
    matches = re.findall(r"\|(acc|word_perplexity)\s*\|\s*([\d.]+)\s*\|", result.stdout, re.DOTALL)
    print(matches)
    if len(matches)<1:
        # 从最新的 result_*.json 文件中提取 acc
        latest_file = get_latest_result_file(ckp_name,args)
        for task in datasets:
            acc = get_acc_from_result_file(latest_file, task)
            if acc is not None:
                with open(f'eval_result/{model_name}/{task}_{model_name}.jsonl', 'a+') as fp:
                    line = json.dumps(dict(ckp_name=ckp_name, acc=acc[-1]))
                    print(line)
                    fp.write(line + '\n')
    else:
        for i, task in enumerate(tasks):
            with open(f'eval_result/{model_name}/{task}_{model_name}.jsonl', 'a+') as fp:
                match = matches[i]
                acc = float(match)
                line = json.dumps(dict(ckp_name=ckp_name, acc=acc))
                print(line)
                fp.write(line + '\n')

    print('time: ', time.time() - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # /data3/liuyang/checkpoints/e_line
    # /data3/liuyang/checkpoints/xl_f_line
    parser.add_argument("--checkpoints_dir", type=str, default="/data3/liuyang/checkpoints/xl_f_line/")
    parser.add_argument("--model_name", type=str, default="xl")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizers/xmodel/v11/xmodel_65280")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = args.device
    model_name = args.model_name
    # model_path = os.path.join(args.checkpoints_dir, args.model_name)
    model_path = args.checkpoints_dir
    print('model_name: ' + str(model_name))

    # load data
    # datasets = ['arc_challenge', 'arc_easy', 'boolq', 'openbookqa', 'piqa', 'winogrande']

    # if model_name in ['nano', 'micro']:
    #     datasets = ['arc_easy', 'hellaswag', 'piqa']
    # elif model_name in ['tiny', 'small', 'medium']:
    
    
#     datasets = ['agieval_lsat_ar', 'arc_ar', 'hellaswag_ar', 'm_mmlu_ar', 'xstorycloze_ar', 'arc_fr', 
#  'french_bench_wikitext_fr', 'hellaswag_fr', 'lambada_openai_mt_fr', 'm_mmlu_fr', 
#  'mgsm_direct_fr', 'mgsm_en_cot_fr', 'mgsm_native_cot_fr', 'paws_fr', 'xwinograd_fr', 
#  'mgsm_direct_th', 'mgsm_en_cot_th', 'mgsm_native_cot_th', 'xcopa_th',"xnli_th",
# ,'belebele_tha_Thai', 'arc_zh', 'xcopa_zh', 'xnli_zh', 'ceval-valid'
# ]
    datasets = ['hellaswag_ar', 'm_mmlu_ar', 'arc_ar', 'arc_fr','french_bench_wikitext_fr','hellaswag_fr',  'xcopa_th',"xnli_th",
'belebele_tha_Thai', 'arc_zh', 'xcopa_zh', 'xnli_zh']
    # datasets = [ 'm_mmlu_ar','french_bench_wikitext_fr','belebele_tha_Thai', 'arc_zh', 'xcopa_zh', 'xnli_zh']
    # , 'ceval-valid' 这是个group,不能一起match
    print(datasets)
    # load tokenizer

    while True:

        dirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
        dirs = sorted(dirs)
        # dirs = [dirs[i*5] for i in range(len(dirs)//5)]

        filename = f'eval_result/{model_name}/{datasets[0]}_{model_name}.jsonl'
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

        try:

            for ckp_name in tqdm(dirs):

                ckp_folder = os.path.join(model_path, ckp_name)
                deploy_model(ckp_folder, model_name, tokenizer_path=args.tokenizer_path)

                for eval_name in datasets:
                    if not os.path.exists(f'eval_result/{model_name}'):
                        os.makedirs(f'eval_result/{model_name}', exist_ok=True)

                print(f'eval_{model_name} <- {ckp_name}')
                print(f'eval_{model_name} <- {ckp_name}')
                latest_file = get_latest_result_file(ckp_name,args)
                if latest_file:
                    # 如果有结果文件，则跳过 do_task
                    for task in datasets:
                        acc = get_acc_from_result_file(latest_file, task)
                        if acc is not None:
                            with open(f'eval_result/{model_name}/{task}_{model_name}.jsonl', 'a+') as fp:
                                line = json.dumps(dict(ckp_name=ckp_name, acc=acc[-1]))
                                print(line)
                                fp.write(line + '\n')
                else:
                    # 否则执行任务
                    do_task(datasets, ckp_name)
                # do_task(datasets,ckp_name)

        except subprocess.CalledProcessError:
            time.sleep(5)
