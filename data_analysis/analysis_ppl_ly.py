import argparse
import os
import pathlib

import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('--source', type=str, default='ZWJCYLK', help='')
    parser.add_argument('--language', type=str, default="['zh', 'en']", help='')
    args = parser.parse_args()

    # source = 'arxiv'
    source = args.source
    # language = ['zh']
    language = eval(args.language)
    print('language: ' + str(language))

    index = 0
    dict_ppl = dict()
    file_path = 'data_analysis/%s/lang_ppl' % source
    for file in os.listdir(file_path):
        with open(os.path.join(file_path, file), 'r') as f:
            for line in f:
                temp = line.split('\t')
                lang, tok, ppl = temp[0], int(temp[1]), float(temp[2])
                if lang not in dict_ppl.keys():
                    dict_ppl[lang] = [[], []]
                dict_ppl[lang][0].append(ppl)
                dict_ppl[lang][1].append(tok)
                index += 1
                if index % 1000000 == 0:
                    print(index)
    print('dataload OK')

    for lan in language:
        data = dict_ppl[lan]
        plt.figure(figsize=(16, 6))
        plt.hist(data[0], weights=data[1], bins=100, range=(0, 5000))
        # histogram1d(data[0], weights=data[1], bins=100, range=(0,5000))
        # sns.kdeplot(data[0], weights=data[1])
        plt.xlabel('Perplexity')
        plt.ylabel('Tokens')
        folder = 'data_analysis/%s/png' % source
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{folder}/ppl_tok_{lan}.png')
