import argparse
import os

import numpy as np
import torch
import transformers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_version", type=str, default='v6', help="tokenizer version")
    parser.add_argument("--vocab_size", type=int, default=32000, help="vocab size")
    args = parser.parse_args()

    data = np.memmap(os.path.join('data/in_domain', f'in_domain_{args.tokenizer_version}_{args.vocab_size}.bin'),
                     dtype=np.uint16, mode='r')

    max_length = 2048
    batch_size = 15

    ix = torch.randint(len(data) - max_length, (batch_size,))
    batch = torch.stack([torch.from_numpy((data[i:i + max_length]).astype(np.int64)) for i in ix])

    print('batch: ' + str(batch))
    print('batch.shape: ' + str(batch.shape))

    new_batch = torch.ones(size=(batch_size, max_length), dtype=torch.long) * -100

    for i in range(batch_size):
        drawing_mode = False
        for j in range(max_length):
            # 看到 "人工发送：" 即开始
            if j <= max_length - 3 and batch[i, j] == 6963 and batch[i, j + 1] == 7522 and batch[i, j + 2] == 392:
                drawing_mode = True

            if drawing_mode:
                new_batch[i, j] = batch[i, j]

            # if batch[i, j] == 97:
            #     drawing_mode = False
            # 看到 "买家发送：" 或 "机器人发送：" 即结束
            if (j <= max_length - 3 and batch[i, j] == 97 and batch[i, j + 1] == 6233) or (
                    j <= max_length - 3 and batch[i, j] == 97 and batch[i, j + 1] == 7213):
                drawing_mode = False

    print('new_batch: ' + str(new_batch))
    print('new_batch.shape: ' + str(new_batch.shape))

    new_batch = new_batch.numpy()

    tokenizer = transformers.AutoTokenizer.from_pretrained('tokenizers/xmodel/v6/xmodel_32000', trust_remote_code=True)
    tokens = new_batch[0]
    tokens[tokens == -100] = 0
    text = tokenizer.decode(tokens)

    print('text: ' + str(text))
