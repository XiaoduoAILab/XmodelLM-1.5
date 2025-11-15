import json
import os

import matplotlib.pyplot as plt

if __name__ == "__main__":
    folder = 'result'
    files = [f for f in os.listdir(folder) if f.endswith('.jsonl')]

    for f in files:
        file_path = os.path.join(folder, f)
        model_name = f[5:-6]

        with open(file_path, 'r') as fp:
            lines = fp.readlines()

        data = []
        for line in lines:
            item = json.loads(line)
            data.append(item)

        data.sort(key=lambda item: item['num_iter'])

        x = [item['num_iter'] for item in data]
        y = [item['accuracy'] for item in data]

        plt.plot(x, y, label=model_name)

    plt.xlabel('steps')
    plt.xscale('log')
    plt.ylabel('MMLU')
    plt.legend()
    plt.show()
