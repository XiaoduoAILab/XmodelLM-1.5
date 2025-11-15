import argparse
import hashlib
import os

from tqdm import tqdm


def calculate_checksum(file_path):
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)

    with open(file_path, 'rb', buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
        return h.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default='/data2/liuyang/redpajama-data-1T/', help="")
    args = parser.parse_args()

    base_url = 'https://data.together.xyz/redpajama-data-1T/v1.0.0/'

    checksum_files = [file for file in os.listdir('sha256') if file.endswith('.txt')]
    print(checksum_files)

    # folders = ['_'.join(file.split('_')[:-1]) for file in checksum_files]
    # print(folders)

    lines = []
    for file in checksum_files:
        filename = os.path.join('sha256', file)
        with open(filename, 'r') as fp:
            lines += fp.readlines()

    missing = []
    corrupted = []
    for line in tqdm(lines):
        line = line.strip()
        tokens = line.split('  ')
        checksum_expected = tokens[0]
        file_name = tokens[1]
        file_path = os.path.join(args.data_path, file_name)
        file_url = os.path.join(base_url, file_name)
        if not os.path.exists(file_path):
            missing.append(file_url)
            continue

        checksum_real = calculate_checksum(file_path)
        if checksum_expected != checksum_real:
            corrupted.append(file_url)

    print('missing: \n' + '\n'.join(missing))
    print('\n\n\n')
    print('corrupted: \n' + '\n'.join(corrupted))

    with open('missing.txt', 'w') as fp:
        missing = [line + '\n' for line in missing]
        fp.writelines(missing)

    with open('corrupted.txt', 'w') as fp:
        missing = [line + '\n' for line in corrupted]
        fp.writelines(corrupted)
