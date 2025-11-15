import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from timeout_decorator import timeout, TimeoutError
from utils import *

'''
raw csv to preprocessed json
'''


def preprocess_safe(text):
    try:
        @timeout(0.01)
        def preprocess(text):
            return del_spec(msg_sub(text))
        processed_text = preprocess(text)
        return processed_text
    except TimeoutError:
        print("preprocess_safe time out!")
        return ''


def process(file_path, output_path):
    '''
    2.6GB (1795w lines): 6min
    '''
    try:
        df = pd.read_csv(file_path, names=['snick', 'cnick', 'time', 'role', 'msg'])
    except:
        print(f'read csv error: {file_path}')
        return 0
    text = []
    snick = cnick = dialog = ''
    for i in range(len(df)):
        if df['role'][i] not in ['人工', '机器人', '买家']:
            continue
        if df['snick'][i] != snick or df['cnick'][i] != cnick:
            dialog = preprocess_safe(dialog)
            if dialog != '':
                text.append({'text': dialog})
            dialog = ''
            snick, cnick = df['snick'][i], df['cnick'][i]
        dialog += '%s发送：%s\n'%(df['role'][i], df['msg'][i])
    with open(output_path, 'w', encoding='utf-8') as f:
        num = 0
        for line in text:
            num += 1
            json.dump(line, f, ensure_ascii=False)
            f.write('\n')
    return num


def process_dir(input_dir, output_dir, src, start, end):
    files = sorted(os.listdir(input_dir))[start:end]
    files.sort()
    nums = []
    for file in tqdm(files):
        date = file.split('_')[-1][:-4]
        file_path, output_path = os.path.join(input_dir, file), os.path.join(output_dir, src, '%s.json'%(date))
        nums.append(process(file_path, output_path))
    print(sum(nums))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--input_dir', type=str, default='/data3/wangyichuan/domain_data/data_raw')
    parser.add_argument('--ouptput_dir', type=str, default='/data3/wangyichuan/domain_data/data_preprocess')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=10000)
    parser.add_argument('--src', type=str, default='ks')
    parser.add_argument('--year', type=int, default=2024)
    args = parser.parse_args()

    input_path, output_path, input_dir, output_dir, start, end, src, year = args.input_path, args.output_path, args.input_dir, args.ouptput_dir, args.start, args.end, args.src, args.year
    input_dir = os.path.join(input_dir, str(year), src)
    output_dir = os.path.join(output_dir, str(year))

    # input_path, output_path = '/data/wangyichuan/domain_data/data_raw/2024/jd/gj_jd_snick_20240307.csv', '/data/wangyichuan/domain_data/data_preprocess/jd_20240307.json'

    if input_path is not None:
        num = process(input_path, output_path)
    else:
        process_dir(input_dir, output_dir, src, start, end)
