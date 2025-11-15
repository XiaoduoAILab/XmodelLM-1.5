import json

filename = 'result/estimate_loss.jsonl'

with open(filename, 'r') as fp:
    lines = fp.readlines()

result = []
for i, line in enumerate(lines[:99]):
    item = json.loads(line)
    item['num_iter'] = (i+1) * 100
    item['model_name'] = 'nano'
    result.append(json.dumps(item) + '\n')

for i, line in enumerate(lines[99:198]):
    item = json.loads(line)
    item['num_iter'] = (i+1) * 100
    item['model_name'] = 'micro'
    result.append(json.dumps(item) + '\n')

for i, line in enumerate(lines[198:297]):
    item = json.loads(line)
    item['num_iter'] = (i+1) * 100
    item['model_name'] = 'tiny'
    result.append(json.dumps(item) + '\n')

for i, line in enumerate(lines[297:396]):
    item = json.loads(line)
    item['num_iter'] = (i+1) * 100
    item['model_name'] = 'small'
    result.append(json.dumps(item) + '\n')

print(result)
with open('result/estimate_loss_fixed.jsonl', 'w') as fp:
    fp.writelines(result)