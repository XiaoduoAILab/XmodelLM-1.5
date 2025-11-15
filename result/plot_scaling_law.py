import matplotlib.pyplot as plt
import json
from math import log10

with open('result/estimate_loss_fixed.jsonl', 'r') as fp:
    lines = fp.readlines()


flops_per_token = dict()
flops_per_token['nano'] = 146291328
flops_per_token['micro'] = 324433152
flops_per_token['tiny'] = 561564672
flops_per_token['small'] = 1257619968

data = dict()
data['nano'] = [], []
data['micro'] = [], []
data['tiny'] = [], []
data['small'] = [], []

for line in lines:
    item = json.loads(line)
    model_name = item['model_name']
    num_iter = item['num_iter']
    val_loss = item['val_loss']
    ops = log10(flops_per_token[model_name] * 983040 * num_iter)
    data[model_name][0].append(ops)
    data[model_name][1].append(val_loss)


for key in data.keys():
    plt.plot(data[key][0], data[key][1], label=key)

plt.legend()
plt.show()




