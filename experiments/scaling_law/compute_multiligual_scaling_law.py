import json
import os

import matplotlib.pyplot as plt
import numpy as np

# 定义路径
base_dir = "/home/wangqun/f_line/eva_result"
result_files = sorted([os.path.join(base_dir, d, "results.json") for d in os.listdir(base_dir) if
                       os.path.isdir(os.path.join(base_dir, d))])
# 定义一个空列表来存储表格数据
table_data = []

# 定义每步的 token 数（假设一个固定值）
tokens_per_step = 4096 * 3 * 40 * 7  # 根据你的示例中的 batch size 和其他参数计算

# 定义一个空列表来存储表格数据
table_data = []
x = []
y = []
# 遍历每个 results.json 文件
for result_file in result_files:
    checkpoint_name = os.path.basename(os.path.dirname(result_file))
    if not os.path.exists(result_file):
        continue

    with open(result_file, "r") as f:
        data = json.load(f)

    # 获取 results 部分
    results = data.get("results", {})

    # 提取 xnli_th, xnli_fr, xnli_ar 的 acc 和 acc_stderr
    row = [checkpoint_name]
    acc_values = []
    # for task in ["xnli_th", "xnli_fr", "xnli_ar"]
    task = "xnli_ar"
    if task in results:
        acc = results[task].get("acc,none", 0)
        acc_stderr = results[task].get("acc_stderr,none", 0)
        # 计算步数
        step_number = int(checkpoint_name.split("iter-")[1])
        tokens_processed = step_number * tokens_per_step

        x.append(tokens_processed)
        y.append(-np.log(acc))

logx = np.log(x)
logy = np.log(y)

coeffs = np.polyfit(logx, logy, deg=1)
poly = np.poly1d(coeffs)
yfit = lambda x: np.exp(poly(np.log(x)))

# 预测 step_number=600000 时的 y 值
predicted_tokens = 600000 * tokens_per_step
predicted_y = yfit(predicted_tokens)
print(predicted_y)
# 泰语 0.9981628536312938
# 法语 0.7576490155746259
# 阿拉伯语 1.076933492062902
plt.loglog(x, yfit(x), 'r--')
plt.xlabel('Data')
plt.ylabel('Test Loss')
plt.title(f"F-line-multiligual-scaling-law-fit_{task}")
plt.loglog(x, y, marker="^", c='blue')
plt.legend()
plt.savefig(f'/home/wangqun/f_line_evaluation_prediction_{task}.png')
# plt.show()
print("done")

# 定义每步的 token 数（假设一个固定值）
tokens_per_step = 4096 * 3 * 40 * 7  # 根据你的示例中的 batch size 和其他参数计算

# 定义不同的 phase 值
phases = [44000, 190000, 600000]

# 定义要遍历的任务列表
# "xnli_th", "xnli_fr", "xnli_ar",  # /home/wangqun/compute_multiligual_scaling_law.py
tasks = ["xcopa_th", "arc_ar", "arc_fr"]

# 存储所有任务的预测值
results = {task: [] for task in tasks}

for phase in phases:  # 遍历任务列表
    for task in tasks:
        result_file = f"/data2/wangqun/llm_pretrain_xmodel/eval_result/xl/{task}_xl.jsonl"

        # 读取指定的 results.jsonl 文件
        with open(result_file, "r") as f:
            data = [json.loads(line) for line in f.readlines()]

        x = []
        y = []

        for entry in data:
            step_number = int(entry["ckp_name"].split("iter-")[1])

            # 仅处理在当前 phase 之前的步骤
            if step_number > phase:
                break

            acc = entry.get("acc", 0)

            # 处理 acc 为 0 的情况，赋值为一个极小的数
            if acc == 0:
                acc = np.finfo(float).eps

            tokens_processed = step_number * tokens_per_step

            x.append(tokens_processed)
            y.append(-np.log(acc))

        # 对数变换并拟合
        logx = np.log(x)
        logy = np.log(y)
        coeffs = np.polyfit(logx, logy, deg=1)
        poly = np.poly1d(coeffs)
        yfit = lambda x: np.exp(poly(np.log(x)))

        # 预测 step_number=600000 时的 y 值
        predicted_tokens = 600000 * tokens_per_step
        predicted_y = yfit(predicted_tokens)

        # 保存结果
        results[task].append(predicted_y)

        # 绘图
        plt.figure()
        plt.loglog(x, yfit(x), 'r--', label="Fitted Line")
        plt.loglog(x, y, marker="^", c='blue', label="Data")
        plt.xlabel('Tokens Processed')
        plt.ylabel('Test Loss (log scale)')
        plt.title(f"F-line-multiligual-scaling-law-fit_{task}_before_{phase}")
        plt.legend()

        # 保存图像
        save_path = f'/data2/wangqun/llm_pretrain_xmodel/experiments/scaling_law/pngs/f_line_evaluation_prediction_{task}_before_{phase}.png'
        plt.savefig(save_path)
        plt.close()  # 关闭当前图表
        print(f"Done, saved at {save_path}")
        print(task, phase, predicted_y)

# 统一绘制任务的预测值图表
plt.figure(figsize=(10, 6))
phases_labels = ['Phase 1', 'Phase 2', 'Phase 3']
for task, values in results.items():
    plt.plot(phases_labels, values, marker='o', label=task)

# 设置图表属性
plt.xlabel('Phase')
plt.ylabel('Predicted Test Loss')
plt.title('Predicted Test Loss for Different Tasks and Phases')
plt.legend()
plt.grid(True)

# 保存图表
summary_save_path = '/data2/wangqun/llm_pretrain_xmodel/experiments/scaling_law/pngs/predicted_test_loss_summary.png'
plt.savefig(summary_save_path)
plt.show()

print(f"Summary plot saved at {summary_save_path}")
