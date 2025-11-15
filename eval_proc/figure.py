import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def plot(js_path):
    fig, axs = plt.subplots(len(js_files), 1, sharex=True, sharey=False, figsize=(12, len(js_files) * 6))
    for index, js_file in enumerate(js_files):
        ds = '_'.join(js_file.split('_')[:-1])

        with open(os.path.join(js_path, js_file), 'r') as f:
            data = f.readlines()
            keys = list(json.loads(data[0]).keys())
            if 'task' in keys:
                keys.remove('task')
            x, y = [], []
            for item in data:
                item = json.loads(item)
                num_iter = int(item[keys[0]][-6:])
                if num_iter not in x:
                    x.append(num_iter)
                    y.append(item[keys[1]])
            axs[index].plot(np.log10(x), y)
            # axs[index].plot(x, y)
            axs[index].set_ylabel(keys[1])
            axs[index].set_title(js_file[:-6] + '    acc - log(step)')

    plt.savefig(js_path)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="eval_result/e_line")
    parser.add_argument("--model_name", type=str, default="xl")
    args = parser.parse_args()

    ds = ['lambada', 'ptb', 'wikitext', 'lambada_acc', 'ptb_acc', 'wikitext_acc', 'winograd', 'wmt', 'coqa', 'mteb_cls',
          'mteb_sts']
    ds = ['arc_challenge', 'arc_easy', 'boolq', 'hellaswag', 'openbookqa', 'piqa', 'sciq', 'winogrande']
    # ds = ['arc_easy', 'hellaswag', 'piqa']

    js_files = [d + '_%s.jsonl' % args.model_name for d in ds]
    js_path = os.path.join(args.result_path, args.model_name)
    plot(js_path)
