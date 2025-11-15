import matplotlib.pyplot as plt
import numpy as np

x = [2288064, 9147264, 24125952, 74337024]
y = [0, 0.0015, 0.0061, 0.0265]
logx = np.log(x)
logy = np.log(y)

coeffs = np.polyfit(logx, logy, deg=1)
poly = np.poly1d(coeffs)
yfit = lambda x: np.exp(poly(np.log(x)))
plt.loglog(x, yfit(x), 'r--')

plt.xlabel('Parameters')
plt.ylabel('log(gsm8k)')
plt.loglog(x, y, marker="s", c='blue')
plt.show()
