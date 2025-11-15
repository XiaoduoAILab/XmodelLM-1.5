import numpy as np
from matplotlib import pyplot as plt
# curve-fit() function imported from scipy
from scipy.optimize import curve_fit


# Test function with coefficients as parameters
def test(x, a, b):
    return a * x + b


if __name__ == "__main__":
    x = [72563328,
         176977152,
         364956672,
         962707968]
    x = np.array(x)
    x = np.log10(x)

    x = x * 983040 * 9900

    y = [4.52875, 4.2351953125, 3.993525390625, 3.888486328125]
    y = np.log10(y)

    param, _ = curve_fit(test, x, y)
    print(param)

    xx = np.linspace(min(x), max(x), num=100)
    yy = param[0] * xx + param[1]
    plt.plot(xx, yy)

    plt.scatter(x, y, marker="^")
    plt.show()
