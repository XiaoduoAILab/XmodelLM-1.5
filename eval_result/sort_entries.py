import argparse
import os
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--line", type=str, default='e_line', help="lines")
    parser.add_argument("--model_name", type=str, default='xl', help="model name")

    args = parser.parse_args()

    line = args.line

    folder = f'eval_result/{line}/{args.model_name}'
    files = [f for f in os.listdir(folder) if f.endswith('.jsonl')]

    for f in files:
        filename = f'eval_result/{line}/{args.model_name}/{f}'
        with open(filename, 'r') as f:
            data = f.readlines()
            data = [json.loads(line) for line in data]
            data = sorted(data, key=lambda x: x['ckp_name'])
            lines = [json.dumps(item) + '\n' for item in data]

        with open(filename, 'w') as f:
            f.writelines(lines)
