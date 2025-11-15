import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from models.modeling_xmodel import XModelConfig, XModelForCausalLM

if __name__ == "__main__":
    config = XModelConfig.from_name('xl')
    config.vocab_size = 32000
    model = XModelForCausalLM(config)
    model.cuda()
    print(model)

    num_tests = 1000
    inputs = torch.randint(0, config.vocab_size, size=(1, 128)).to(model.device)

    start = time.time()
    for _ in tqdm(range(num_tests)):
        outputs = model(inputs)

    end = time.time()
    elapsed = end - start
    print('elapsed per forward: {:.2f} ms'.format(elapsed))
