import argparse
import os
import pickle
import random

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer

if __name__ == "__main__":
    num_sent = 10000
    seed = 42
    block_size = 512
    rng = random.Random(seed)

    data_config = [
        # 英文
        ("arxiv", 0.0137),
        ("book", 0.0257),
        ("c4", 0.0857),
        ("cc", 0.4672),
        ("github", 0.0129),
        ("starcoder", 0.0459),
        ("stackexchange", 0.0149),
        ("wiki", 0.0446),

        ("amps", 0.0003),
        ("Bookcorpus", 0.0005),
        ("fanfics", 0.0017),
        ("openwebmath", 0.0069),
        ("pubmed", 0.0086),

        # 中文
        ("law", 0.0086),
        ("news", 0.0043),
        ("patent", 0.0043),
        ("webtext", 0.0552),
        ("PTD", 0.0562),

        # 领域
        ("in_domain", 0.1428)
    ]
    datasets = [item[0] for item in data_config]
    weights = [float(item[1]) for item in data_config]

    tokenizer = AutoTokenizer.from_pretrained('tokenizers/xmodel/v6/xmodel_32000/', trust_remote_code=True)
    embed_model = SentenceTransformer('/data2/wangyichuan/full_inspection/inspection/model/pretrain/bge-base-zh-v1.5')

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/home/liuyang/ds_bins/', help="data path")
    args = parser.parse_args()

    data_path = args.data_path

    filenames = []
    passages = []
    for _ in tqdm(range(num_sent)):
        prefix = rng.choices(datasets, weights=weights, k=1)[0]
        files = [f for f in os.listdir(data_path) if f.startswith(prefix) and f.endswith('.bin')]
        file = random.choices(files)[0]
        filenames.append(file)

        filename = os.path.join(data_path, file)
        data = np.memmap(filename, dtype=np.uint16, mode='r')
        i = random.randint(0, len(data) - block_size - 1)
        tokens = data[i:i + block_size]
        text = tokenizer.decode(tokens)
        passages.append(text)

    embeddings = embed_model.encode(passages, normalize_embeddings=True)

    data = dict(files=filenames, passages=passages, embeddings=embeddings)
    with open('embed.pkl', 'wb') as fp:
        pickle.dump(data, fp)
