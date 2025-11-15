import argparse
import os

import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('--source', type=str, default='ZWJCYLK', help='')
    args = parser.parse_args()

    source = args.source

    lang_ins, lang_tok, total_tok = dict(), dict(), 0
    lid_tok_path = f'data_analysis/{source}/lang_tok'
    for file in [file for file in os.listdir(lid_tok_path) if file != 'lang_tok.txt']:
        with open(os.path.join(lid_tok_path, file), 'r') as f:
            for line in f:
                line = line.replace('\n', '').split('\t')
                lang, ins, tok = line[0][:-1], int(line[2]), int(line[3])
                if lang not in lang_tok.keys():
                    lang_ins[lang], lang_tok[lang] = 0, 0
                lang_ins[lang] += ins
                lang_tok[lang] += tok
                total_tok += tok
    with open(os.path.join(lid_tok_path, 'lang_tok.txt'), 'w') as f:
        for lang in lang_ins.keys():
            f.write('%s \t%s \t%s \t%.2f\n' % (lang, lang_ins[lang], lang_tok[lang], lang_tok[lang] / total_tok * 100))

    l = list(lang_tok.items())[:20]
    l.sort(key=lambda x: x[1], reverse=True)
    plt.figure(figsize=(20, 6))
    plt.bar([i[0][:-1] for i in l], [i[1] for i in l], width=0.5, log=True)
    plt.xlabel('Language')
    plt.ylabel('Tokens')
    plt.savefig(f'data_analysis/{source}/png/lang_tok.png')

    l = list(lang_ins.items())[:20]
    l.sort(key=lambda x: x[1], reverse=True)
    plt.figure(figsize=(20, 6))
    plt.bar([i[0][:-1] for i in l], [i[1] for i in l], width=0.5, log=True)
    plt.xlabel('Language')
    plt.ylabel('Instances')
    plt.savefig(f'data_analysis/{source}/png/lang_ins.png')
