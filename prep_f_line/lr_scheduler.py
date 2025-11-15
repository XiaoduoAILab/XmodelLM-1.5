import math

import matplotlib.pyplot as plt


def get_lr(it):
    start_stable = 412000
    end_stable = 480000
    lr_stable = 1.5e-4
    half_life = 40000

    if start_stable <= it <= end_stable:
        return lr_stable

    if it > end_stable:
        lr = lr_stable * 0.5 ** ((it - end_stable) / half_life)
        lr = max(min_lr, lr)
        return lr

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


warmup_iters = 2000
learning_rate = 6e-4
min_lr = 2e-5
lr_decay_iters = 600000

iter_num = range(lr_decay_iters)
lr = [get_lr(it) for it in iter_num]

plt.plot(iter_num, lr)
plt.show()
