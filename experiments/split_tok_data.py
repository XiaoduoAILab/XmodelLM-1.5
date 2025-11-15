import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model')
    # general
    parser.add_argument('--data_path', type=str, default='/data4/mishiqian/data/msg.txt', help='')
    args = parser.parse_args()

    train_ratio = 0.7

    with open(args.data_path, 'r') as fp:
        lines = fp.readlines()

    num_samples = len(lines)
    num_train = int(num_samples * train_ratio)

    train = lines[:num_train]
    test = lines[num_train:]

    with open('train.txt', 'w') as fp:
        fp.writelines(train)

    with open('test.txt', 'w') as fp:
        fp.writelines(test)
