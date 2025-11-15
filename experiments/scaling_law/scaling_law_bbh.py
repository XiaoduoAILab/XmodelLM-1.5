import matplotlib.pyplot as plt
import numpy as np

x = [2288064, 9147264, 24125952, 74337024]
y = [0.0025, 0.0283, 0.0746, 0.1564]
logx = np.log(x)
logy = np.log(y)

coeffs = np.polyfit(logx, logy, deg=1)
poly = np.poly1d(coeffs)
yfit = lambda x: np.exp(poly(np.log(x)))
plt.loglog(x, yfit(x), 'r--')

plt.xlabel('Parameters')
plt.ylabel('log(bbh)')
plt.loglog(x, y, marker="s", c='blue')
plt.show()
