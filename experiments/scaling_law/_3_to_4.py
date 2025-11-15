import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

num_params = {'nano': 2288064, 'micro': 9147264, 'tiny': 24125952, 'small': 74337024, 'medium': 270582784,
              'large': 594617856, 'xl': 594617856}
eps = 1e-8


# Formula 1.5 in https://arxiv.org/pdf/2001.08361.pdf
def func(x, n_c, d_c, a_n, a_d):
    n, d = x

    return ((n_c / n) ** (a_n / a_d) + d_c / d) ** a_d


if __name__ == "__main__":

    # model_name_list = ['nano', 'micro', 'tiny']
    model_name_list = ['nano', 'micro']
    data = []

    for model_name in model_name_list:
        filename = f'data/run-{model_name}_redpajama_run1-tag-Main_val_loss.json'
        with open(filename, 'r') as fp:
            points = json.load(fp)

        for p in points:
            if p[1] > 0:
                data.append([num_params[model_name], p[1], p[2]])

    print(data)
    print(len(data))

    arr = np.array(data)
    print('arr.shape: ' + str(arr.shape))
    xdata = np.vstack((arr[:, 0], arr[:, 1]))
    print('xdata.shape: ' + str(xdata.shape))

    ydata = arr[:, 2]
    print('ydata.shape: ' + str(ydata.shape))

    original = (1, 1, 1, 1)

    popt, pcov = curve_fit(func, xdata, ydata)
    print('fitted: ' + str(popt))



    plt.plot(xdata[1], func(xdata, *popt), 'g--',
             label='fit: n_c=%5.3f, d_c=%5.3f, a_n=%5.3f, a_d=%5.3f' % tuple(popt))


    plt.xlabel('data')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
