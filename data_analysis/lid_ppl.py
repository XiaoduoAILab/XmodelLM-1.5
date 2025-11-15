import argparse
import json
import os
import sys
import time

# sys.path.append('/data/wangyichuan')
from analysis.tokenizer import count_seq_len_sp
from analysis.lid import get_lang
from analysis.perplexity import get_ppl

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('--begin', type=int, default='0', help='')
    parser.add_argument('--end', type=int, default='5', help='')
    parser.add_argument('--interval', type=int, default='1000', help='')
    parser.add_argument('--data_path', type=str, default='/data2/liuyang/redpajama-data-1T/arxiv', help='')
    parser.add_argument('--result_path', type=str, default='/data/wangyichuan/data_analysis/arxiv/lang_ppl/', help='')
    args = parser.parse_args()

    begin, end, interval = args.begin, args.end, args.interval
    result_path = args.result_path

    # data_path = '/data2/liuyang/redpajama-data-1T/arxiv'
    data_path = args.data_path
    files = os.listdir(data_path)

    result = []
    start = time.time()
    index = 0
    for i in range(begin, end):
        with open(os.path.join(data_path, files[i]), 'r', encoding='utf-8') as f:
            for line in f:
                index += 1
                if index % interval == 0:
                    print(index, (time.time() - start) / 60)
                text = json.loads(line)['text']
                lang, tok_len = get_lang(text), count_seq_len_sp(text)
                ppl = get_ppl(text=text, lang=lang)
                result.append((lang, tok_len, ppl))

    with open(result_path + 'lang_ppl_%d_%d.txt' % (begin, end), 'w') as f:
        for item in result:
            f.write('%s\t%s\t%s\n' % (item[0], item[1], item[2]))

    print('finish')
