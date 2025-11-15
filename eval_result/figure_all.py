import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def draw_graph(f):
    task = f.replace('.jsonl', '')
    filename = os.path.join(folder, f)

    with open(filename, 'r') as f:
        data = f.readlines()

    keys = list(json.loads(data[0]).keys())
    x, y = [], []
    for item in data:
        item = json.loads(item)
        x.append(int(item[keys[0]][-6:]))
        y.append(item[keys[1]])

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    save_folder = f'eval_result/graph_{args.line}'
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # print(x)
    ax.plot(x, y)
    plt.title(f'{args.line} task: {task}')
    plt.savefig(f'{save_folder}/{task}_raw.jpg')
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    tokens_per_iteration = 4096 * 120 * 7
    x = x * tokens_per_iteration
    # x = np.log(x)
    eps = 1e-8
    y[y == 0] = eps
    y = -np.log(y / 100)

    # print('x: ' + str(x))
    # print('y: ' + str(y))

    ax.set_xscale('log')
    ax.plot(x, y)
    plt.xlabel('train data')
    plt.ylabel('-log(acc)')
    plt.title(f'{args.line} task: {task}')
    plt.savefig(f'{save_folder}/{task}_log_log.jpg')
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--line", type=str, default='e_line', help="model name")
    parser.add_argument("--model_name", type=str, default='xl', help="model name")

    args = parser.parse_args()

    folder = f'eval_result/{args.line}/{args.model_name}'
    files = [f for f in os.listdir(folder) if f.endswith('.jsonl')]

    for f in tqdm(files):
        try:
            draw_graph(f)
        except IndexError as err:
            print(err)
