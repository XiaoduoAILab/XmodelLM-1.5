import os
import json
import time

import fasttext
from model.model import *


ppl_model = KenlmModel('/data1/redpajama-v2-process/process/model', 'en')
lid_model = fasttext.load_model('/data1/redpajama-v2-process/process/model/lid.176.bin')

sent_len_threshold = 50
punct_ratio_threshold = 0.1
ppl_threshold = 300


def get_ppl(text):
    return ppl_model.get_perplexity(text)


def get_lang(text):
    text = text.replace('\n', '')
    pred = lid_model.predict(text, k=1)
    if pred[1][0]>0.5:
        return pred[0][0][-2:]
    else:
        return 'not identified'


def del_spec(s):
    spec = ["</s>", "<!DOCTYPE html>", "<html>", "<head>", "<title>", "</head>", "<body>", "<h1>", "<h2>", "<h3>", "<h4>", "<h5>", "<h6>",
            "<p>", "<a>", "<strong>", "<em>", "<br>", "<span>", "<ul>", "<ol>", "<li>", "<table>", "<tr>", "<td>", "<th>", "<form>", "<input>",
            "<textarea>", "<select>", "<button>", "<img>", "<audio>", "<video>", "<style>", "<script>", "<div>", "<span>", "<hr>"]
    for sp in spec:
        s = s.replace(sp, ' ')
    return s


def cal_punct_cond(s, punct_ratio_threshold=punct_ratio_threshold):
    punct_marks = '''!()-[]{};:'"\,<>./?@#$%^&*_~！（）《》【】；：‘’“”，。？——'''
    count = 0
    for char in s:
        if char in punct_marks:
            count += 1
    return count / (len(s)+0.01) > punct_ratio_threshold


def cal_sentLen_cond(s, lang, sent_len_threshold=sent_len_threshold):
    if lang in ['zh']:
        return len(s) < sent_len_threshold
    elif lang in ['en', 'fr', 'ru', 'de', 'es', 'it', 'uk', 'pl', 'pt', 'nl', 'sv', 'ca', 'cs', 'bg', 'da', 'no', 'fi', 'el', 'az']:
        return len(s.split(' ')) < sent_len_threshold
    return len(s) < sent_len_threshold * 2



if __name__ == '__main__':
    text = 'i am the boss, you are not.'
    print(get_lang(text), get_ppl(text))