import math
import matplotlib.pyplot as plt

def get_lr(it):
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


if __name__ == "__main__":
    warmup_iters = 2000
    learning_rate = 3e-4
    min_lr = 1e-5
    lr_decay_iters = 700000

    x = list(range(lr_decay_iters))
    y = [get_lr(it) for it in x]

    plt.plot(x, y)

    key_x = [350000+i*350000 for i in range(2) ]
    key_y = [get_lr(it) for it in key_x]

    print(key_x, key_y)

    plt.scatter(key_x, key_y, marker='^')

    plt.show()