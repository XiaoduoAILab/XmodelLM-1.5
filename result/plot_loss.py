import matplotlib.pyplot as plt
import json

with open('result/estimate_loss_fixed.jsonl', 'r') as fp:
    lines = fp.readlines()

data = dict()
data['nano'] = []
data['micro'] = []
data['tiny'] = []
data['small'] = []

for line in lines:
    item = json.loads(line)
    model_name = item['model_name']
    num_iter = item['num_iter']
    val_loss = item['val_loss']
    data[model_name].append(val_loss)

x = range(100, 10000, 100)
for key in data.keys():
    plt.plot(x, data[key], label=key)

plt.legend()
plt.show()
