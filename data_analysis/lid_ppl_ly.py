import argparse
import json
import os
import time

import sentencepiece as spm
from tqdm import tqdm

from analysis.lid import get_lang
from analysis.perplexity import get_ppl
from analysis.tokenizer import count_seq_len_sp


def extract_text(json_obj):
    if 'text' in json_obj:
        return json_obj['text']
    elif 'Content' in json_obj:
        return json_obj['Content']
    elif 'content' in json_obj:
        return json_obj['content']
    elif 'data' in json_obj:
        return json_obj['data']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('--begin', type=int, default='0', help='')
    parser.add_argument('--end', type=int, default='5', help='')
    parser.add_argument('--interval', type=int, default='1000', help='')
    parser.add_argument('--data_path', type=str, default='/data2/liuyang/redpajama-data-1T/arxiv', help='')
    parser.add_argument('--result_path', type=str, default='data_analysis/arxiv/lang_ppl/', help='')
    parser.add_argument('--tokenizer', type=str, default='tokenizers/xmodel/v6/xmodel_32000/xmodel_32000.model',
                        help='tokenizer')
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)

    begin, end, interval = args.begin, args.end, args.interval
    result_path = args.result_path

    # data_path = '/data2/liuyang/redpajama-data-1T/arxiv'
    data_path = args.data_path
    files = os.listdir(data_path)

    result = []
    start = time.time()
    index = 0
    for f in files:
        print(f)
        with open(os.path.join(data_path, f), 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                index += 1
                # if index % interval == 0:
                #     print(index, (time.time() - start) / 60)
                text = extract_text(json.loads(line))
                lang, tok_len = get_lang(text), count_seq_len_sp(text, sp)
                ppl = get_ppl(text=text, lang=lang)
                result.append((lang, tok_len, ppl))

    with open(result_path + 'lang_ppl_%d_%d.txt' % (begin, end), 'w') as f:
        for item in result:
            f.write('%s\t%s\t%s\n' % (item[0], item[1], item[2]))

    print('finish')
