import re
import json
import os
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
from tqdm import tqdm

def remove_repeated_patterns(text):
    pattern = r"(.+?)(\1{2,})"
    result = re.sub(pattern, r"\1", text)
    result = re.sub(r'\n+', '\n', result)
    result = result.strip('\n')

    return result


def remove_repeated_lines(text):
    lines = text.split('\n')
    unique_lines = []
    seen_lines = set()
    
    for line in lines:
        line = line.strip()
        if line and line not in seen_lines:
            unique_lines.append(line)
            seen_lines.add(line)
    
    return '\n'.join(unique_lines)

def remove_repeat(text):
    text = remove_repeated_patterns(text)
    text = remove_repeated_lines(text)
    return text


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def content_sim(s1, s2):
    similarity = torch.nn.functional.cosine_similarity(s1, s2).item()
    return similarity

def findan(text):
    answer_start = text.find("Answer:") 
    answer_content = text[answer_start + len("Answer: "):].strip()
    return answer_content

def do_json(lines):
    uesd_file = './th_en_mt_group10_single_0.xlsx'
    df = pd.read_excel(uesd_file)
    flag = 1
    choices = ['B', 'C', 'D']
    for line in tqdm(lines, desc="Processing lines"):
        e1 = get_embedding(line['A'])
        e2 = get_embedding(line['B'])
        e3 = get_embedding(line['C'])
        e4 = get_embedding(line['D'])
        co1 = content_sim(e2, e1)
        if co1 > 0.9:
            flag = (flag + 1) % 50
            line['B'] = findan(df['th_model_translate_output'].iloc[flag])
        co2 = content_sim(e1, e4)
        if co2 > 0.9:
            flag = (flag + 1) % 50
            line['D'] = findan(df['th_model_translate_output'].iloc[flag])
        co3 = content_sim(e1, e3)
        if co3 > 0.9:
            flag = (flag + 1) % 50
            line['C'] = findan(df['th_model_translate_output'].iloc[flag])
        co4 = content_sim(e2, e4)
        if co4 > 0.9:
            flag = (flag + 1) % 50
            line['B'] = findan(df['th_model_translate_output'].iloc[flag])
        co5 = content_sim(e2, e3)
        if co5 > 0.9:
            flag = (flag + 1) % 50
            line['B'] = findan(df['th_model_translate_output'].iloc[flag])
        co6 = content_sim(e3, e4)
        if co6 > 0.9:
            flag = (flag + 1) % 50
            line['C'] = findan(df['th_model_translate_output'].iloc[flag])        
        for choice in choices:
            line[choice] = remove_repeat(line[choice])
    return lines

def main():
    file_path = './multiple_choice_task_4_2.json'
    output_file = './processed4.json'


    if os.path.isfile(file_path):
        output_file = './processed' + str(int(output_file[11]) + 1) + '.json'

    with open(file_path) as fp:
        lines = json.load(fp)

    lines = do_json(lines)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(lines, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    main()
