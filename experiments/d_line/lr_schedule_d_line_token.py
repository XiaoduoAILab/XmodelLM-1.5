import matplotlib.pyplot as plt

if __name__ == "__main__":
    warmup_iters = 2000
    learning_rate = 3e-4
    min_lr = 1e-5
    lr_decay_iters = 625000

    x = [0, 1081344000000]
    y = [2048, 2048]

    x.append(x[-1])
    y.append(4096)
    x.append(x[-1] + 98304000000)
    y.append(4096)

    x.append(x[-1])
    y.append(8192)
    x.append(x[-1] + 98304000000)
    y.append(8192)

    x.append(x[-1])
    y.append(16384)
    x.append(x[-1] + 98304000000)
    y.append(16384)

    plt.ylim(0, 20000)
    plt.plot(x, y, c='blue')

    plt.ylabel('Sequence length')
    plt.xlabel('Tokens')
    plt.title('D-Line Ladder')
    plt.show()
