import matplotlib.pyplot as plt
from transformers import AdamW, get_cosine_schedule_with_warmup

from models.modeling_xmodel import XModelForCausalLM, XModelConfig

if __name__ == "__main__":
    config = XModelConfig.from_name('nano')
    model = XModelForCausalLM(config)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(model_parameters, lr=3e-4, betas=(0.9, 0.95), eps=1e-6,
                      weight_decay=0.1)

    num_warmup_steps = 2000
    num_training_steps = 10000

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                   num_warmup_steps=num_warmup_steps,
                                                   num_training_steps=num_training_steps)
    print(lr_scheduler)
    x = []
    y = []

    for i in range(num_training_steps):
        lr_scheduler.step()
        x.append(i)
        y.append(lr_scheduler.get_lr())

    plt.plot(x, y)
    plt.show()
