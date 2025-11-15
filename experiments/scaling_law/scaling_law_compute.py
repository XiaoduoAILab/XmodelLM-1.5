import matplotlib.pyplot as plt
import numpy as np

x = [1.43810e+18, 3.18931e+18, 5.52041e+18, 1.23629e+19, 3.39263e+19, 5.28841e+19, 8.75820e+19]
y = [3.747, 3.206, 2.956, 2.720, 2.589, 2.491, 2.443]
logx = np.log(x)
logy = np.log(y)

coeffs = np.polyfit(logx, logy, deg=1)
poly = np.poly1d(coeffs)
yfit = lambda x: np.exp(poly(np.log(x)))
plt.loglog(x, yfit(x), 'r--')

plt.xlabel('Compute')
plt.ylabel('Test Loss')
plt.loglog(x, y, marker="^", c='blue')
plt.show()
