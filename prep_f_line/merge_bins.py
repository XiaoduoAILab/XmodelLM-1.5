import numpy as np

if __name__ == "__main__":
    dy_path = '/data1/liuruoyu/data_bins/dy/dy.bin'
    ks_path = '/data1/liuruoyu/data_bins/ks/ks.bin'
    tb_path = '/data1/liuruoyu/data_bins/tb/tb.bin'

    dy_data = np.memmap(dy_path, dtype=np.uint16, mode='r')
    ks_data = np.memmap(ks_path, dtype=np.uint16, mode='r')
    tb_data = np.memmap(tb_path, dtype=np.uint16, mode='r')

    len_dy = len(dy_data)
    len_ks = len(ks_data)
    len_tb = len(tb_data)

    print('len_dy: ' + str(len_dy))
    print('len_ks: ' + str(len_ks))
    print('len_tb: ' + str(len_tb))

    total_len = len_dy + len_ks + len_tb

    val_len = 5000000
    train_len = total_len - val_len

    train_arr = np.memmap('/home/liuyang/ds_bins/haixin.bin', dtype=np.uint16, mode='w+', shape=(train_len,))
    train_arr[0:len_dy] = dy_data[:]
    train_arr[len_dy:len_dy + len_ks] = ks_data[:]
    train_arr[len_dy + len_ks:] = tb_data[:-val_len]
    train_arr.flush()

    val_arr = np.memmap('/data1/liuyang/llm_pretrain_xmodel/data/haixin/haixin_test.bin', dtype=np.uint16, mode='w+',
                        shape=(val_len,))
    val_arr[:] = tb_data[-val_len:]
    val_arr.flush()
