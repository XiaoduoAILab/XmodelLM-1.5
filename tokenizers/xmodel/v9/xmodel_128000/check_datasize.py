import os
import pandas as pd
from tqdm import tqdm

# 数据源配置，这里我们使用字典以便于通过键名访问
data_config = {
    "arxiv": 0.0160,
    "book": 0.0300,
    "c4": 0.1000,
    "cc": 0.5451,
    "github": 0.0150,
    "starcoder": 0.0536,
    "stackexchange": 0.0174,
    "wiki": 0.0520,
    "amps": 0.0003,
    "Bookcorpus": 0.0006,
    "fanfics": 0.0020,
    "openwebmath": 0.0080,
    "pubmed": 0.0100,
    "law": 0.0100,
    "news": 0.0050,
    "patent": 0.0050,
    "webtext": 0.0644,
    "PTD": 0.0656,
}

files_path = "/data/wangqun/v8_corpus_0"
source_sizes = []


def get_total_size_of_txt_files(directory,keys):
    total_size = 0
    for source in tqdm(keys, desc="Processing source"):
        source_path = os.path.join(directory, source)  # 构造数据源路径
        
        if os.path.exists(source_path):
            # 遍历子目录
            for root, dirs, files in os.walk(source_path):
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
        else:
            # 没有子目录，直接在主目录中查找匹配的文件
            pattern = f"{source}.txt"
            for file in os.listdir(directory):
                if file == pattern:
                    file_path = os.path.join(directory, file)
                    if os.path.isfile(file_path):
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
        source_sizes.append({'source': source, 'File Size': file_size, 'File Size (read)': size_transform(file_size)})

    return total_size # sum(size['File Size'] for size in source_sizes)


def size_transform (file_size):
    file_size_mb = file_size / (1024 * 1024)  # 转换为兆字节
    file_size_gb = file_size / (1024 * 1024 * 1024)  # 转换为吉字节
    if file_size_gb < 1:
        file_size_formatted = "{:.2f} MB".format(file_size_mb)
    else:
        file_size_formatted = "{:.2f} GB".format(file_size_gb)
        
    return file_size_formatted
# 调用函数并计算大小
total_txt_size = get_total_size_of_txt_files(files_path,data_config.keys())
print(f"Total size of .txt files in '{files_path}': {size_transform(total_txt_size)} ")

# 创建DataFrame
df_sizes = pd.DataFrame(source_sizes)
# 保存到Excel文件
excel_path = os.path.join(files_path, 'eline_file_sizes_v8_tokenizer_test_data.xlsx')
df_sizes.to_excel(excel_path, index=False)
print(f"File size information saved to {excel_path}")
