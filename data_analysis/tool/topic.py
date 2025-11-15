import os
import time
import json
import torch
import fasttext
import numpy as np
from scipy.special import expit



num = 10
# data_path = '/data2/liuyang/redpajama-data-1T/c4/c4-train.00000-of-01024.jsonl'
data_path = '/data2/liuyang/redpajama-data-1T/book/book.jsonl'


model_path = '/data/wangyichuan/process/analysis/model/topic.bin'
model = fasttext.load_model(model_path)
def get_topic(text, model=model):
    text = text.replace('\n', '')
    result = model.predict(text)
    label, prob = result[0][0][9:], result[1][0]
    if prob > 0.1:
        return label
    else:
        return 'Null'


if __name__ == '__main__':
    start = time.time()
    index = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = json.loads(line)['text'].replace('\n', '')
            topic = get_topic(text)
            index += 1
            if index == num:
                break
    end = time.time()
    print(end-start)