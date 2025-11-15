import os
import matplotlib.pyplot as plt


src = 'arxiv'

dic_topic = dict()
lid_tok_path = '/data/wangyichuan/data_analysis/%s/lang_topic'%src
for file in [file for file in os.listdir(lid_tok_path) if file != 'lang_topic.txt']:
    with open(os.path.join(lid_tok_path, file), 'r') as f:
        for line in f:
            line = line.replace('\n','').split('\t')
            topic, ins = line[0], int(line[1])
            if topic not in dic_topic.keys():
                dic_topic[topic] = 0
            dic_topic[topic] += ins

with open(os.path.join(lid_tok_path, 'lang_topic.txt'), 'w') as f:
    for topic in dic_topic.keys():
        f.write('%s\t%d\n'%(topic, dic_topic[topic]))


l = list(dic_topic.items())[:15]
l.sort(key=lambda x:x[1], reverse=True)
plt.figure(figsize=(20, 6))
plt.bar([i[0] for i in l], [i[1] for i in l], width=0.5, log=True)
plt.xlabel('Topic')
plt.ylabel('Instances')
plt.xticks(rotation=10, fontsize=10)
plt.savefig('/data/wangyichuan/data_analysis/%s/png/topic.png'%src)
