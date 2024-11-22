import json
import random

data_path = '../data/xdata_thai_test_v3.json'
choices = ['A', 'B', 'C', 'D']

with open(data_path, 'r', encoding='utf-8') as fp:
    data = json.load(fp)

examples = []
for item in data:
    question = item['question']
    correct = item['A']
    options = [item['A'], item['B'], item['C'], item['D']]
    random.shuffle(options)
    index = options.index(correct)
    answer = choices[index]

    examples.append(dict(question=question, options=options, answer=answer))

num_examples = len(examples)
print('num_examples: ' + str(num_examples))

num_dev = 8
num_test = 350

dev = examples[:num_dev]
test = examples[num_dev:]

with open('../data/xdata_thai_dev_v3.json', 'w') as fp:
    for example in dev:
        json_line = json.dumps(example) + '\n'
        fp.write(json_line)

with open('../data/xdata_thai_test_v3.json', 'w') as fp:
    for example in test:
        json_line = json.dumps(example) + '\n'
        fp.write(json_line)
