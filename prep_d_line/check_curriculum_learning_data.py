import numpy as np
import transformers

if __name__ == '__main__':
    num_seg = 10
    chunk_size = 256
    data = np.memmap('/data1/llm_pretrain_xmodel/prep_d_line_data/data.bin', dtype=np.uint16, mode='r')
    length = len(data)
    print('length: ' + str(length))
    mid = length // 2
    print('mid: ' + str(mid))
    print(data[0:chunk_size])

    tokenizer = transformers.AutoTokenizer.from_pretrained('tokenizers/llama/')
    print(tokenizer)

    # print('begin ======================================================================')
    # text = tokenizer.decode(data[:chunk_size])
    # print(text)
    #
    # print('middle ======================================================================')
    # text = tokenizer.decode(data[mid:mid + chunk_size])
    # print(text)
    #
    # print('end ======================================================================')
    # text = tokenizer.decode(data[-chunk_size:])
    # print(text)

    for i in range(num_seg):
        print(f'segment {i} ======================================================================')
        start = i * length // num_seg
        text = tokenizer.decode(data[start:start + chunk_size])
        print(text)
