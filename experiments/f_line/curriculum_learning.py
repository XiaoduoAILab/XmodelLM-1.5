import matplotlib.pyplot as plt


def get_weight(it, idx):
    return start[idx] + (end[idx] - start[idx]) * (it / num_iters)


if __name__ == "__main__":
    start = [1.0, 0.0, 0.0]
    end = [0.85, 0.1, 0.05]
    num_iters = 600 * 1000

    x = list(range(num_iters))
    y_0 = [get_weight(it, 0) for it in x]
    y_1 = [get_weight(it, 1) for it in x]
    y_2 = [get_weight(it, 2) for it in x]

    plt.plot(x, y_0, color='red', label='English')
    plt.plot(x, y_1, color='green', label='Chinese')
    plt.plot(x, y_2, color='blue', label='Others')

    plt.xlabel('steps')
    plt.ylabel('weight')
    plt.title('Curriculum Learning')
    plt.legend()
    plt.show()
