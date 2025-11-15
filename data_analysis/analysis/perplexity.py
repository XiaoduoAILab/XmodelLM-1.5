import os
import json
import time
from kenlm import LanguageModel

import sys  
# sys.path.append('/data/wangyichuan')
from analysis.model import *


num = 1000000
data_path = '/data2/liuyang/redpajama-data-1T/c4/c4-train.00000-of-01024.jsonl'
# data_path = '/data2/liuyang/redpajama-data-1T/book/book.jsonl'


lm_path = '/data1/liuyang/llm_pretrain_xmodel/kenlm'
ccnet_path = '/data/wangyichuan/cc_net/data/lm_sp'
lm_name = [lm[:2] for lm in os.listdir(lm_path) if lm.endswith('bin')]
ccnet_name = [lm[:2] for lm in os.listdir(ccnet_path) if lm.endswith('bin')]

start = time.time()
af = KenlmModel(lm_path, 'af')
ar = KenlmModel(lm_path, 'ar')
ar = KenlmModel(lm_path, 'ar')
As = KenlmModel(lm_path, 'as')
bn = KenlmModel(lm_path, 'bn')
ca = KenlmModel(lm_path, 'ca')
en = KenlmModel(lm_path, 'en')
es = KenlmModel(lm_path, 'es')
eu = KenlmModel(lm_path, 'eu')
fr = KenlmModel(lm_path, 'fr')
gu = KenlmModel(lm_path, 'gu')
hi = KenlmModel(lm_path, 'hi')
id = KenlmModel(lm_path, 'id')
kn = KenlmModel(lm_path, 'kn')
ml = KenlmModel(lm_path, 'ml')
mr = KenlmModel(lm_path, 'mr')
pt = KenlmModel(lm_path, 'pt')
sw = KenlmModel(lm_path, 'sw')
ta = KenlmModel(lm_path, 'ta')
te = KenlmModel(lm_path, 'te')
ur = KenlmModel(lm_path, 'ur')
vi = KenlmModel(lm_path, 'vi')
yo = KenlmModel(lm_path, 'yo')
zh = KenlmModel(lm_path, 'zh')
az = KenlmModel(ccnet_path, 'az')
be = KenlmModel(ccnet_path, 'be')
bg = KenlmModel(ccnet_path, 'bg')
cs = KenlmModel(ccnet_path, 'cs')
da = KenlmModel(ccnet_path, 'da')
de = KenlmModel(ccnet_path, 'de')
el = KenlmModel(ccnet_path, 'el')
et = KenlmModel(ccnet_path, 'et')
fa = KenlmModel(ccnet_path, 'fa')
fi = KenlmModel(ccnet_path, 'fi')
he = KenlmModel(ccnet_path, 'he')
hr = KenlmModel(ccnet_path, 'hr')
hu = KenlmModel(ccnet_path, 'hu')
hy = KenlmModel(ccnet_path, 'hy')
Is = KenlmModel(ccnet_path, 'is')
it = KenlmModel(ccnet_path, 'it')
ja = KenlmModel(ccnet_path, 'ja')
ka = KenlmModel(ccnet_path, 'ka')
kk = KenlmModel(ccnet_path, 'kk')
km = KenlmModel(ccnet_path, 'km')
ko = KenlmModel(ccnet_path, 'ko')
lt = KenlmModel(ccnet_path, 'lt')
lv = KenlmModel(ccnet_path, 'lv')
mk = KenlmModel(ccnet_path, 'mk')
mn = KenlmModel(ccnet_path, 'mn')
my = KenlmModel(ccnet_path, 'my')
ne = KenlmModel(ccnet_path, 'ne')
nl = KenlmModel(ccnet_path, 'nl')
no = KenlmModel(ccnet_path, 'no')
pl = KenlmModel(ccnet_path, 'pl')
ro = KenlmModel(ccnet_path, 'ro')
ru = KenlmModel(ccnet_path, 'ru')
uk = KenlmModel(ccnet_path, 'uk')


def get_ppl(text, lang, lm_path=lm_path):
    if lang == 'af':
        return af.get_perplexity(text)
    elif lang == 'ar':
        return ar.get_perplexity(text)
    elif lang == 'as':
        return As.get_perplexity(text)
    elif lang == 'az':
        return az.get_perplexity(text)
    elif lang == 'be':
        return be.get_perplexity(text)
    elif lang == 'bg':
        return bg.get_perplexity(text)
    elif lang == 'bn':
        return bn.get_perplexity(text)
    elif lang == 'ca':
        return ca.get_perplexity(text)
    elif lang == 'cs':
        return cs.get_perplexity(text)
    elif lang == 'da':
        return da.get_perplexity(text)
    elif lang == 'de':
        return de.get_perplexity(text)
    elif lang == 'el':
        return el.get_perplexity(text)
    elif lang == 'en':
        return en.get_perplexity(text)
    elif lang == 'es':
        return es.get_perplexity(text)
    elif lang == 'et':
        return et.get_perplexity(text)
    elif lang == 'eu':
        return eu.get_perplexity(text)
    elif lang == 'fa':
        return fa.get_perplexity(text)
    elif lang == 'fi':
        return fi.get_perplexity(text)
    elif lang == 'fr':
        return fr.get_perplexity(text)
    elif lang == 'gu':
        return gu.get_perplexity(text)
    elif lang == 'he':
        return he.get_perplexity(text)
    elif lang == 'hi':
        return hi.get_perplexity(text)
    elif lang == 'hr':
        return hr.get_perplexity(text)
    elif lang == 'hu':
        return hu.get_perplexity(text)
    elif lang == 'hy':
        return hy.get_perplexity(text)
    elif lang == 'id':
        return id.get_perplexity(text)
    elif lang == 'is':
        return Is.get_perplexity(text)
    elif lang == 'it':
        return it.get_perplexity(text)
    elif lang == 'ja':
        return ja.get_perplexity(text)
    elif lang == 'ka':
        return ka.get_perplexity(text)
    elif lang == 'kk':
        return kk.get_perplexity(text)
    elif lang == 'km':
        return km.get_perplexity(text)
    elif lang == 'kn':
        return kn.get_perplexity(text)
    elif lang == 'ko':
        return ko.get_perplexity(text)
    elif lang == 'lt':
        return lt.get_perplexity(text)
    elif lang == 'lv':
        return lv.get_perplexity(text)
    elif lang == 'mk':
        return mk.get_perplexity(text)
    elif lang == 'ml':
        return ml.get_perplexity(text)
    elif lang == 'mn':
        return mn.get_perplexity(text)
    elif lang == 'mr':
        return mr.get_perplexity(text)
    elif lang == 'my':
        return my.get_perplexity(text)
    elif lang == 'ne':
        return ne.get_perplexity(text)
    elif lang == 'nl':
        return nl.get_perplexity(text)
    elif lang == 'no':
        return no.get_perplexity(text)
    elif lang == 'pl':
        return pl.get_perplexity(text)
    elif lang == 'pt':
        return pt.get_perplexity(text)
    elif lang == 'ro':
        return ro.get_perplexity(text)
    elif lang == 'ru':
        return ru.get_perplexity(text)
    elif lang == 'sw':
        return sw.get_perplexity(text)
    elif lang == 'ta':
        return ta.get_perplexity(text)
    elif lang == 'te':
        return te.get_perplexity(text)
    elif lang == 'uk':
        return uk.get_perplexity(text)
    elif lang == 'ur':
        return ur.get_perplexity(text)
    elif lang == 'vi':
        return vi.get_perplexity(text)
    elif lang == 'yo':
        return yo.get_perplexity(text)
    elif lang == 'zh':
        return zh.get_perplexity(text)
    else:
        return 1e5


if __name__ == '__main__':
    print('ppl model loaded!: ', time.time()-start)
    lm_name.sort()
    ccnet_name.sort()
    all_name = sorted(list(set(lm_name).union(set(ccnet_name))))
    # for i in sorted(lm_name):
    #     print("%s = KenlmModel(lm_path, '%s')"%(i,i))
    # for i in sorted([lm for lm in ccnet_name if lm not in lm_name]):
    #     print("%s = KenlmModel(ccnet_path, '%s')"%(i,i))
    # for i in all_name:
    #     print("    elif lang == '%s':"%i)
    #     print('        return %s.get_perplexity(text)'%i)
    start = time.time()
    index = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = json.loads(line)['text']
            ppl = get_ppl(text, lang='en')
            index += 1
            if index == num:
                break
    end = time.time()
    print(end-start)
