import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt


def load_data(file):
    filename = os.path.join(folder, file)
    with open(filename, 'r') as f:
        data = f.readlines()
    keys = list(json.loads(data[0]).keys())
    x, y = [], []
    for item in data:
        item = json.loads(item)
        x.append(int(item[keys[0]][-6:]))
        y.append(item[keys[1]])

    return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--line", type=str, default='f_line', help="model name")
    parser.add_argument("--model_name", type=str, default='xl', help="model name")
    parser.add_argument("--task", type=str, default='arithmetic')
    args = parser.parse_args()

    folder = f'eval_result/{args.line}/{args.model_name}/'
    files = [f for f in os.listdir(folder) if f.endswith('.jsonl') and f.startswith(args.task)]

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)

    for f in files:
        x, y = load_data(f)
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        tokens_per_iteration = 4096 * 120 * 7
        x = x * tokens_per_iteration
        eps = 1e-8
        y[y == 0] = eps
        y = -np.log(y / 100)
        ax.plot(x, y, label=f)

    ax.set_xscale('log')
    plt.legend()
    plt.show()
