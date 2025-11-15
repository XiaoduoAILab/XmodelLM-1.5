import random

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def b_line():
    data_size = 8592031756 // 256
    bin_file = np.zeros((data_size,))
    block_size = 2048 // 256
    num_tests = 600000 * 240

    for _ in tqdm(range(num_tests)):
        idx = random.randrange(len(bin_file) - block_size)
        bin_file[idx:idx + block_size] += 1
        # for i in range(idx, idx + block_size):
        #     bin_file[i] += 1

    plt.plot(bin_file)
    plt.show()

    print(np.max(bin_file))
    print(np.min(bin_file))
    print(len(bin_file) - np.count_nonzero(bin_file))


def c_line():
    data_size = int(1.4e12 // 1024)
    bin_file = np.zeros((data_size,))
    block_size = 4096 // 1024
    num_tests = 350000 * 960

    for _ in tqdm(range(num_tests)):
        idx = random.randrange(len(bin_file) - block_size)
        bin_file[idx:idx + block_size] += 1
        # for i in range(idx, idx + block_size):
        #     bin_file[i] += 1

    plt.plot(bin_file)
    plt.show()

    print(np.max(bin_file))
    print(np.min(bin_file))
    print(len(bin_file))
    print(np.count_nonzero(bin_file))
    print(np.count_nonzero(bin_file) / len(bin_file))


if __name__ == "__main__":
    c_line()
