import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
from fast_histogram import histogram1d

source = 'arxiv'
language = ['en']

index = 0
dict_ppl = dict()
file_path = '/data/wangyichuan/data_analysis/%s/lang_ppl'%source
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
    plt.hist(data[0], weights=data[1], bins=100, range=(0,5000))
    # histogram1d(data[0], weights=data[1], bins=100, range=(0,5000))
    # sns.kdeplot(data[0], weights=data[1])
    plt.xlabel('Perplexity')
    plt.ylabel('Tokens')
    plt.savefig('/data/wangyichuan/data_analysis/%s/png/ppl_tok_%s.png'%(source,lan))