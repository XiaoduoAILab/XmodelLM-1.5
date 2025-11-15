import json

import matplotlib.pyplot as plt
import numpy as np

log_name = 'tb_logs/run-xl_openwebtext_run2-tag-Main_val_loss.json'
data = json.loads(open(log_name, 'r').read())
print(len(data))

x = []
y = []

i = 0
while True:
    x.append(data[2 ** i][1])
    y.append(data[2 ** i][2])
    i += 1
    if 2 ** i >= len(data):
        break

logx = np.log(x)
logy = np.log(y)

coeffs = np.polyfit(logx, logy, deg=1)
poly = np.poly1d(coeffs)
yfit = lambda x: np.exp(poly(np.log(x)))
plt.loglog(x, yfit(x), 'r--')
plt.xlabel('Data')
plt.ylabel('Test Loss')
plt.title(log_name.split('-')[1])
plt.loglog(x, y, marker="^", c='blue')
plt.show()
