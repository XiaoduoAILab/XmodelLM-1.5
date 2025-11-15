import math

import matplotlib.pyplot as plt


def get_lr(it):
    if it <= 500000:
        learning_rate = 6e-4
        min_lr = 6e-5
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)
    else:
        cont_steps = 500000
        max_steps = 600000
        learning_rate = 6e-5
        min_lr = 1e-5
        a = -(learning_rate - min_lr) / (max_steps - cont_steps)
        b = learning_rate - a * cont_steps
        return a * it + b


if __name__ == "__main__":
    warmup_iters = 2000
    learning_rate = 6e-4
    min_lr = 6e-5
    lr_decay_iters = 500000
    max_steps = 600000

    x = list(range(max_steps))
    y = [get_lr(it) for it in x]

    plt.plot(x, y)

    # key_x = [250000+i*250000 for i in range(4) ]
    # key_y = [get_lr(it) for it in key_x]
    #
    # print(key_x, key_y)
    #
    # plt.scatter(key_x, key_y, marker='^')

    plt.show()
