import numpy as np

if __name__ == "__main__":
    filename_1 = '/data2/liuyang/lit-redpajama-new/book_0000000000.bin'
    filename_2 = 'book.bin'

    HDR_SIZE = 24
    data = np.memmap(filename_1, dtype=np.uint16, mode='r', offset=HDR_SIZE, order="C")
    arr_len = 1024 ** 2 * 8192
    arr = np.memmap(filename_2, dtype=np.uint16, mode='w+', shape=(arr_len,))
    arr[0:arr_len] = data[0:arr_len]
    arr.flush()
