from datasets import load_dataset

if __name__ == "__main__":
    filename = '/data2/data_zh/culturaX/zh/zh_part_00000.parquet'
    dataset = load_dataset('parquet', data_files={'test': filename})

    num_example = 199
    with open('data/culturax_zh/culturax_zh.txt', 'w') as fp:
        for i, example in enumerate(dataset['test']):
            print(i)
            text = example['text']
            print(type(text))
            print(text)
            fp.write(text + '\n\n')
            if i > num_example:
                break
