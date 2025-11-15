import argparse
import json
import struct

import numpy as np
import transformers
from tqdm import tqdm


def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()

    parser.add_argument("--bin_path", type=str, default="/data2/liuyang/lit-redpajama-new/arxiv_0000000000.bin",
                        help="bin path")

    return parser


if __name__ == "__main__":
    dtypes = {
        1: np.uint8,
        2: np.int8,
        3: np.int16,
        4: np.int32,
        5: np.int64,
        6: np.float32,
        7: np.float64,
        8: np.uint16,
    }

    HDR_MAGIC = b"LITPKDS"
    HDR_SIZE = 24  # bytes

    chunk_size: int = 4097 * 1024
    doc_size = 4097

    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    with open(args.bin_path, "rb") as f:
        magic = f.read(len(HDR_MAGIC))
        print('magic: ' + str(magic))
        assert magic == HDR_MAGIC, "File doesn't match expected format."

        version = struct.unpack("<Q", f.read(8))
        print('version: ' + str(version))
        assert (1,) == version

        (dtype_code,) = struct.unpack("<B", f.read(1))
        dtype = dtypes[dtype_code]
        print('dtype: ' + str(dtype))

        (chunk_size,) = struct.unpack("<Q", f.read(8))
        print('chunk_size: ' + str(chunk_size))

    mmap = np.memmap(args.bin_path, mode="r", dtype=dtype, offset=HDR_SIZE, order="C")
    print('len(mmap): ' + str(len(mmap)))

    tokenizer = transformers.AutoTokenizer.from_pretrained('tokenizers/llama',
                                                           use_fast=False,
                                                           trust_remote_code=True)

    output_filename = args.bin_path.split('.')[0] + '.json'
    with open(output_filename, 'w') as fp:
        for i in tqdm(range(0, len(mmap), doc_size)):
            doc = mmap[i:i + doc_size]
            text = tokenizer.decode(doc)
            # print('text: ' + str(text))
            json_str = json.dumps(dict(text=text))
            fp.write(json_str + '\n')
