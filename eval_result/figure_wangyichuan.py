import os
import json
import numpy as np
import matplotlib.pyplot as plt


result_path = '/data/wangyichuan/llm_pretrain_xmodel/eval_result/c_line'
model_name = 'small'

gpt_eval_path = '/data/wangyichuan/llm_pretrain_xmodel/gpt/eval'
gpt_eval = 'gpt2.jsonl'


with open(os.path.join(gpt_eval_path, gpt_eval), 'r') as f:
    data_gpt = f.readlines()
add_gpt = False

ds = ['lambada', 'ptb', 'wikitext', 'lambada_acc', 'ptb_acc', 'wikitext_acc', 'winograd', 'wmt', 'coqa', 'mteb_cls', 'mteb_sts']
ds = ['arc_challenge', 'arc_easy', 'boolq', 'openbookqa', 'piqa', 'winogrande']
# ds = ['coqa']
js_files = [d+'_%s.jsonl'%model_name for d in ds]

def plot(js_path):
    fig, axs = plt.subplots(len(js_files), 1, sharex=True, sharey=False, figsize=(12,len(js_files)*6))
    for index, js_file in enumerate(js_files):
        ds = '_'.join(js_file.split('_')[:-1])

        with open(os.path.join(js_path, js_file), 'r') as f:
            data = f.readlines()
            keys = list(json.loads(data[0]).keys())
            x, y = [], []
            for item in data:
                item = json.loads(item)
                x.append(int(item[keys[0]][-6:]))
                y.append(item[keys[1]])
            axs[index].plot(np.log10(x), y)
            axs[index].set_ylabel(keys[1])
            axs[index].set_title(js_file[:-6]+'    acc - log(step)')

            if add_gpt:
                horizontal_line_value = 0
                for item_gpt in data_gpt:
                    item_gpt = json.loads(item_gpt.replace('\n',''))
                    if item_gpt['eval_name'] == ds:
                        horizontal_line_value = item_gpt[keys[1]]
                axs[index].axhline(y=horizontal_line_value, color='green', linestyle='--', label='GPT')

    plt.savefig(js_path)


if __name__ == '__main__':
    js_path = os.path.join(result_path, model_name)
    plot(js_path)