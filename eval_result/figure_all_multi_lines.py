import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def get_hlines():
    with open('eval_result/hlines.jsonl', 'r') as f:
        data = f.readlines()
        data = [json.loads(line) for line in data]

    return data


def get_data(filename):
    with open(filename, 'r') as f:
        data = f.readlines()
        data = [json.loads(line) for line in data]

    data = sorted(data, key=lambda x: x['ckp_name'])

    x, y = [], []
    for item in data:
        x.append(int(item['ckp_name'][-6:]))
        y.append(item['acc'])

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return x, y


def draw_graph(f):
    task = f.replace('.jsonl', '')

    save_folder = f'eval_result/multi_lines'
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    first_line = lines[0]
    read_file = f'eval_result/{first_line}/{args.model_name}/{f}'
    x, y = get_data(read_file)
    ax.plot(x, y, label=first_line)

    # print(x)
    for line in lines[1:]:
        read_file = f'eval_result/{line}/{args.model_name}/{f}'
        if os.path.isfile(read_file):
            x, y = get_data(read_file)
            ax.plot(x, y, label=line)

    task_hlines = [item for item in hlines if item['task'] == task]
    for l in task_hlines:
        plt.axhline(y=l['value'], color='b', linestyle=':', label=l['model'])

    plt.title(f'multi-lines task: {task}')
    plt.legend()
    plt.savefig(f'{save_folder}/{task}_raw.jpg')
    # plt.show()

    tokens_per_iteration = 4096 * 120 * 7
    eps = 1e-8

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    first_line = lines[0]
    read_file = f'eval_result/{first_line}/{args.model_name}/{f}'
    x, y = get_data(read_file)

    x = x * tokens_per_iteration
    x = np.log10(x)
    y[y == 0] = eps
    y = -np.log(y / 100)
    ax.plot(x, y, label=first_line)

    for line in lines[1:]:
        read_file = f'eval_result/{line}/{args.model_name}/{f}'
        if os.path.isfile(read_file):
            x, y = get_data(read_file)
            x = x * tokens_per_iteration
            x = np.log10(x)
            y[y == 0] = eps
            y = -np.log(y / 100)
            ax.plot(x, y, label=line)

        # print('x: ' + str(x))
    # print('y: ' + str(y))

    plt.xlabel('train data')
    plt.ylabel('-log(acc)')
    plt.title(f'multi-lines task: {task}')
    plt.legend()
    plt.savefig(f'{save_folder}/{task}_log_log.jpg')
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lines", type=str, default='f_line,e_line', help="lines")
    parser.add_argument("--model_name", type=str, default='xl', help="model name")

    args = parser.parse_args()

    hlines = get_hlines()

    lines = args.lines
    lines = lines.split(',')

    folder = f'eval_result/{lines[0]}/{args.model_name}'
    files = [f for f in os.listdir(folder) if f.endswith('.jsonl')]

    for f in tqdm(files):
        try:
            draw_graph(f)
        except IndexError as err:
            print(f)
            print(err)
