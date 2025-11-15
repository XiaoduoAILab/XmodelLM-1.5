import sys
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from data_utils import create_dataloaders_curriculum

if __name__ == "__main__":
    micro_batch_size = 15
    max_length = 4096
    sampling = "random"
    seed_offset = 0
    data_path = "/data2/wangqun/f_line_data_v11_65280/"
    vocab_size = 65280

    data_config = [
        # 英文 (0-12)
        ("arxiv", 0.0151),
        ("book", 0.0282),
        ("c4", 0.0941),
        ("cc", 0.5132),
        ("github", 0.0141),
        ("starcoder", 0.0504),
        ("stackexchange", 0.0164),
        ("wiki", 0.0489),

        ("AMPS", 0.0003),
        ("BookCorpus", 0.0006),
        ("fanfics", 0.0019),
        ("Openwebmath", 0.0075),
        ("PubMed", 0.0094),

        # 中文 (13-18)
        ("law", 0.0067),
        ("news", 0.0033),
        ("patent", 0.0033),
        ("webtext", 0.0429),
        ("PTD", 0.0400),
        ("ZWJCYLK", 0.0037),

        # 小语种 (19-20)
        ("culturax", 0.0800),
        ("multiwiki", 0.0200),
    ]

    print(sum([it[1] for it in data_config]))

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    train_dataloader, val_dataloader = create_dataloaders_curriculum(
        batch_size=micro_batch_size,
        block_size=max_length,
        data_config=data_config,
        weights=weights,
        sampling=sampling,
        train_data_dir=data_path,
        val_data_dir=None,
        seed=1337 + seed_offset,
        vocab_size=vocab_size
    )

    print("train_dataloader: " + str(train_dataloader))
    print("train_dataloader.dataset: " + str(train_dataloader.dataset))
    print("train_dataloader.dataset._weights: " + str(train_dataloader.dataset._weights))

    weights[0] = 0.0
    print("train_dataloader.dataset._weights: " + str(train_dataloader.dataset._weights))
