import json

import matplotlib.pyplot as plt


def get_curve(model_name):
    with open(f'{model_name}.jsonl') as fp:
        lines = fp.readlines()

    x = []
    y = []
    prev_iter_num = 1
    for line in lines:
        item = json.loads(line)
        ckp_name = item['ckp_name']

        iter_num = int(ckp_name[5:])
        l2_norm = item['l2_norm']
        datasize = tokens_per_iter * iter_num

        if iter_num / prev_iter_num < 1.2:
            continue

        x.append(datasize)
        y.append(l2_norm)
        prev_iter_num = iter_num

    return x, y


if __name__ == '__main__':
    # with open('../tb_logs/run-large_redpajama_run1-tag-Norm_MAE.json') as fp:

    tokens_per_iter = 3932160
    color_map = dict(nano='gainsboro', micro='bisque', tiny='lavender', small='thistle', medium='lightpink',
                     large='lightgreen', xl='blue')

    plt.figure(figsize=(10, 6))  
    for model_name in ['nano', 'micro', 'tiny', 'small', 'medium', 'large', 'xl']:
        x, y = get_curve(model_name)
        # plt.plot(x, y, marker='^', c=color_map[model_name], label=model_name, markersize=5)
        plt.plot(x, y, c=color_map[model_name], label=model_name)

    plt.xlabel('Dataset Size')
    plt.ylabel('Parameter norms')
    plt.xlim(2e+9, 1e+13)

    plt.xscale('log')
    # plt.yscale('log')

    plt.axvspan(2e+9, 8e+10, facecolor='azure', alpha=0.5)
    plt.axvspan(8e+10, 5e+11, facecolor='ivory', alpha=0.5)
    plt.axvspan(5e+11, 1e+13, facecolor='beige', alpha=0.5)

    plt.text(8e+9, 1900, 'Memorizing \nRegime')
    plt.text(1e+11, 1900, 'Intermediate \nRegime')
    plt.text(1e+12, 1900, 'Generalizing \nRegime')

    # plt.title('Xmodel-Large')
    plt.legend(loc='upper left')
    plt.savefig('norms.pdf')
    plt.show()
