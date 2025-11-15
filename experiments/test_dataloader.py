import sys
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from data_utils import create_dataloaders

if __name__ == "__main__":
    micro_batch_size = 10
    max_length = 4096
    data_path = '/data1/liuyang/lit-redpajama-new/'
    total_steps = 1000

    # Data proportions from https://arxiv.org/pdf/2302.13971.pdf Table 1
    data_config = [
        ("arxiv", 2.5),
        ("book", 4.5),
        ("c4", 15.0),
        ("cc", 67.0),
        ("github", 4.5),
        ("stackexchange", 2.0),
        ("wikipedia", 4.5),
    ]

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=max_length,
        data_config=data_config,
        train_data_dir=data_path,
        val_data_dir=None,
        seed=1337,
    )

    dataloader_iter = iter(train_dataloader)
    data = next(dataloader_iter)  # fetch the very first batch
    data = data.cpu().numpy()

    print('data: ' + str(data))
    print('data.shape: ' + str(data.shape))

    for _ in tqdm(range(total_steps)):
        data = next(dataloader_iter)

    print_text = False
    if print_text:
        tokenizer = AutoTokenizer.from_pretrained('tokenizers/llama')

        for i in range(data.shape[0]):
            text = tokenizer.decode(data[i]).strip()
            print('=============================================')
            print('text: ' + str(text))
            print('=============================================')
