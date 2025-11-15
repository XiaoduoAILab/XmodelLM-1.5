from math import acos

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from tqdm import tqdm


def cos_sim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


if __name__ == "__main__":
    num_test = 1000

    # n_list = list(range(2, 1024))
    # n_list = [2, 3, 5, 10, 20, 100, 1000]
    n_list = [2, 3, 5, 10, 20, 100]
    x_axis = np.arange(0, 3.14, 0.01)

    for n in tqdm(n_list):
        theta_list = []
        for _ in range(num_test):
            a = (np.random.rand(n, ) - 0.5) * 2
            b = (np.random.rand(n, ) - 0.5) * 2
            # print(a, b)

            cosine = cos_sim(a, b)
            theta = acos(cosine)
            # theta = degrees(theta)
            theta_list.append(theta)

        mean = np.mean(theta_list)
        std = np.std(theta_list)

        plt.plot(x_axis, norm.pdf(x_axis, mean, std), label=f'n={n}')

    plt.xlabel('θ')
    plt.ylabel('p(θ)')
    plt.title('PDF')
    plt.legend()
    plt.show()
