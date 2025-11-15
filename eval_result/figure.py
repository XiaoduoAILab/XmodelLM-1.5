import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--line", type=str, default='f_line', help="model name")
    parser.add_argument("--model_name", type=str, default='xl', help="model name")
    parser.add_argument("--task", type=str, default='hellaswag_fr')
    args = parser.parse_args()

    filename = f'eval_result/{args.line}/{args.model_name}/{args.task}.jsonl'

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

    print(x)
    plt.plot(x, y)
    plt.title(f'{args.line} task: {args.task}')
    plt.savefig('eval_result/raw.jpg')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    tokens_per_iteration = 4096 * 120 * 7
    x = x * tokens_per_iteration
    # x = np.log(x)
    eps = 1e-8
    y[y == 0] = eps
    y = -np.log(y / 100)

    print('x: ' + str(x))
    print('y: ' + str(y))

    ax.set_xscale('log')
    ax.plot(x, y)
    plt.xlabel('train data')
    plt.ylabel('-log(acc)')
    plt.savefig('eval_result/log_log.jpg')
    plt.show()
