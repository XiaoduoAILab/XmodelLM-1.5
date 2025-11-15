import matplotlib.pyplot as plt
import numpy as np

x = [2288064, 9147264, 24125952, 74337024, 270582784, 594617856, 1082230784]
y = [3.747, 3.206, 2.956, 2.720, 2.589, 2.491, 2.443]
logx = np.log(x)
logy = np.log(y)

coeffs = np.polyfit(logx, logy, deg=1)
poly = np.poly1d(coeffs)
yfit = lambda x: np.exp(poly(np.log(x)))
plt.loglog(x, yfit(x), 'r--')

plt.xlabel('Parameters')
plt.ylabel('Test Loss')
plt.loglog(x, y, marker="s", c='blue')
plt.show()
