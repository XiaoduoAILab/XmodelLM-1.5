import os

import numpy as np

if __name__ == "__main__":
    data_path = '/data2/liuyang/e_line_v6_32000/'
    files = [f for f in os.listdir(data_path) if f.endswith('.bin')]
    for f in files:
        file_path = os.path.join(data_path, f)
        data = np.memmap(file_path, dtype=np.uint16, mode='r')
        print(f)
        print(np.all(data))
