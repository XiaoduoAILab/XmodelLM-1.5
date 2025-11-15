import argparse
import json
import os
import time

import sentencepiece as spm
from tqdm import tqdm

from analysis.lid import get_lang
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
    parser.add_argument('--source', type=str, default='ZWJCYLK', help='source')
    parser.add_argument('--tokenizer', type=str, default='tokenizers/xmodel/v6/xmodel_32000/xmodel_32000.model',
                        help='tokenizer')
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)

    begin, end, interval = args.begin, args.end, args.interval

    data_path = args.data_path
    source = args.source
    result_path = f'data_analysis/{source}/lang_tok/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    files = os.listdir(data_path)

    lang_ins, lang_tok = dict(), dict()
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
                if lang not in lang_tok.keys():
                    lang_ins[lang], lang_tok[lang] = 0, 0
                lang_ins[lang] += 1
                lang_tok[lang] += tok_len

    with open(result_path + 'lang_tok_%d_%d.txt' % (begin, end), 'w') as f:
        for lang in lang_tok.keys():
            f.write('%s: \t\t%s \t%s \n' % (lang, lang_ins[lang], lang_tok[lang]))

    print('finish')
