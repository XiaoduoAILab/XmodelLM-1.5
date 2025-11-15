model_name = 'medium'

# these make the total batch size be ~0.5M
# 5 batch size * 4096 block size * 3 gradaccum * 8 GPUs = 491,520
batch_size = 3
block_size = 2048
gradient_accumulation_steps = 10 * 8

# adamw optimizer
learning_rate = 3e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 0.5  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla
min_lr = 3e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
