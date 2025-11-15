import os
import gzip
import argparse
from tqdm import tqdm
from util import *


print_del = False

lang_score_threshold = 0.9
sent_len_threshold = 20
punct_ratio_threshold = 0.1
ppl_threshold = 400


def process_single_file(file_path, save_path, n=-1):
    text = []
    num = num_lid = num_punct = num_sent = num_ppl = 0
    with gzip.open(file_path, 'rt', encoding='utf-8') as gz_file:
        for line in gz_file:
            json_data = json.loads(line.strip())
            sent, lang, lang_score, ppl = del_spec(json_data['raw_content']), json_data['language'], json_data['language_score'], json_data['perplexity']
            
            # lid
            if lang != 'en' or lang_score < lang_score_threshold:
                num_lid += 1
                if print_del:
                    print('lid: ', sent)
            # punct ratio
            elif cal_punct_cond(sent, punct_ratio_threshold=punct_ratio_threshold):
                num_punct += 1
                if print_del:
                    print('punct: ', sent)
            # sent len
            elif cal_sentLen_cond(sent, lang, sent_len_threshold=sent_len_threshold):
                num_sent += 1
                if print_del:
                    print('sent len: ', sent)
            # ppl
            elif ppl > ppl_threshold:
                num_ppl += 1
                if print_del:
                    print('ppl: ', sent)
            else:
                text.append(sent)
            num += 1
            if num == n:
                break
    
    with open(save_path, 'a') as f:
        for sent in text:
            f.write(sent + '\n[end of paragraph --wyc]\n')

    return num, len(text), num_lid, num_punct, num_sent, num_ppl


def process_folder(folder_path, save_path):
    file_path = [f for f in os.listdir(folder_path) if f.endswith('gz')]
    num, num_saved, num_lid, num_punct, num_sent, num_ppl = 0, 0, 0, 0, 0, 0
    for f in tqdm(file_path):
        num_temp, num_saved_temp, num_lid_temp, num_punct_temp, num_sent_temp, num_ppl_temp = process_single_file(os.path.join(folder_path, f), save_path)
        num += num_temp
        num_saved += num_saved_temp
        num_lid += num_lid_temp
        num_punct += num_punct_temp
        num_sent += num_sent_temp
        num_ppl += num_ppl_temp
    print('num: %d num_saved: %d(%.4f) num_lid: %d(%.4f) num_punct: %d(%.4f) num_sent: %d(%.4f) num_ppl: %d(%.4f)'%(num, num_saved, num_saved/num, num_lid, num_lid/num, num_punct, num_punct/num, num_sent, num_sent/num, num_ppl, num_ppl/num))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='2014-15', help='')
    args = parser.parse_args()
    folder = args.folder

    folder_path = '/data2/RedPajama-Data-V2/en_head'
    save_folder = '/data1/redpajama-v2-process/data_preprocess'

    save_path = os.path.join(save_folder, folder+'.txt')
    process_folder(os.path.join(folder_path, folder), save_path)