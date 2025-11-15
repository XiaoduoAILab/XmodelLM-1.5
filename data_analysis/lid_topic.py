import os
import numpy as np
import json
import time
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
import sys  
sys.path.append('/data/wangyichuan')
from analysis.lid import get_lang
from analysis.topic import get_topic


data_path = '/data2/liuyang/redpajama-data-1T/arxiv'
files = os.listdir(data_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('--begin', type=int, default='0', help='')
    parser.add_argument('--end', type=int, default='1', help='')
    parser.add_argument('--interval', type=int, default='1000', help='')
    parser.add_argument('--result_path', type=str, default='/data/wangyichuan/data_analysis/arxiv/lang_topic/', help='')
    args = parser.parse_args()

    begin, end, interval = args.begin, args.end, args.interval
    result_path = args.result_path
    end = min(end, len(files))

    
    stat_topic = dict()
    start = time.time()
    index = 0
    for i in range(begin, end):
        with open(os.path.join(data_path, files[i]), 'r', encoding='utf-8') as f:
            for line in f:
                index += 1
                if index % interval == 0:
                    print(index, (time.time()-start)/60)
                text = json.loads(line)['text']
                lang = get_lang(text)
                if lang != 'en':
                    continue
                topic = get_topic(text)
                if topic not in stat_topic.keys():
                    stat_topic[topic] = 0
                stat_topic[topic] += 1


    with open(result_path+'lang_topic_%d_%d.txt'%(begin, end), 'w') as f: 
        for topic in stat_topic.keys():
            f.write('%s\t%s\n'%(topic, stat_topic[topic]))
    
    print('finish')