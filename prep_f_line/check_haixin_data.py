import numpy as np
import transformers

if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained('/data1/liuyang/llm_pretrain_xmodel/tokenizers/xmodel/v6/xmodel_32000/', trust_remote_code=True)
    # train_path = '/home/liuyang/ds_bins/haixin.bin'
    train_path = '/data1/liuruoyu/data_bins/haixin.bin'
    train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
    train_text = tokenizer.decode(train_data[:512])
    print('train_text: ' + str(train_text))

    # val_path = '/data1/liuyang/llm_pretrain_xmodel/data/haixin/haixin_test.bin'
    val_path = '/data1/liuruoyu/data_bins/haixin_test.bin'
    val_data = np.memmap(train_path, dtype=np.uint16, mode='r')
    val_text = tokenizer.decode(val_data[:512])
    print('val_text: ' + str(val_text))

