import json
import scipy
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # dataset = 'arc_challenge'
    # dataset = 'arc_easy'
    # dataset = 'boolq'
    dataset = 'hellaswag'
    # dataset = 'openbookqa'
    # dataset = 'piqa'
    # dataset = 'sciq'
    # dataset = 'winogrande'
    file_path = f'eval_result/e_line/xl/{dataset}_xl.jsonl'

    with open(file_path, 'r') as fp:
        lines = fp.readlines()

    x = []
    y = []
    prev_iter_num = 1
    for line in lines:
        item = json.loads(line)
        ckp_name = item['ckp_name']
        iter_num = int(ckp_name[5:])
        if iter_num < 2000 or iter_num / prev_iter_num < 2:
            continue

        acc = item['acc']
        err = 1 - acc
        x.append(iter_num)
        y.append(err)

        prev_iter_num = iter_num

    logx = np.log(x)
    logy = np.log(y)

    print(f'pearson r={scipy.stats.pearsonr(logx, logy)[0]}')
    print(f'spearman r={scipy.stats.spearmanr(logx, logy)[0]}')

    coeffs = np.polyfit(logx, logy, deg=1)
    poly = np.poly1d(coeffs)
    yfit = lambda x: np.exp(poly(np.log(x)))
    plt.loglog(x, yfit(x), 'r--')

    plt.xlabel('log(iters)')
    plt.ylabel('log(error)')
    plt.loglog(x, y, marker="s", c='blue')
    plt.title(dataset)
    plt.show()
