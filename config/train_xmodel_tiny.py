model_name = 'tiny'

# these make the total batch size be ~0.5M
# 10 batch size * 2048 block size * 3 gradaccum * 8 GPUs = 491,520
batch_size = 15
block_size = 2048
gradient_accumulation_steps = 2 * 8

# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
