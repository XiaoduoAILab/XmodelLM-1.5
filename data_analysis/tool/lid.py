import time
import json
import langid
import fasttext


'''
10000行数据:   langid(20s)  fasttext(0.4s)
100000行数据:  langid(220s)  fasttext(3.6s)
'''


num = 1000
# data_path = '/data2/liuyang/redpajama-data-1T/c4/c4-train.00000-of-01024.jsonl'
data_path = '/data2/liuyang/redpajama-data-1T/book/book.jsonl'

def get_lang1(text):
    return langid.classify(text.replace('\n', ''))[0]

model = fasttext.load_model('/data/wangyichuan/process/analysis/model/lid.176.bin')
def get_lang_prob(text):
    pred = model.predict(text.replace('\n', ''), k=1)
    return (pred[0][0][-2:], pred[1][0])

def get_lang(text):
    text = text.replace('\n', '')
    pred = model.predict(text, k=1)
    if pred[1][0]>0.5:
        return pred[0][0][-2:]
    else:
        return 'not identified'

if __name__ == '__main__':
    start = time.time()
    index = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = json.loads(line)['text']
            lang = get_lang(text)
            lang1 = get_lang(text[:100000])
            if lang != lang1:
                print(lang, lang1, text[:20000])
            index += 1
            if index == num:
                break
    end = time.time()
    print(end-start)