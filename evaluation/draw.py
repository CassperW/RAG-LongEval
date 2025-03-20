import json
import matplotlib.pyplot as plt
import numpy as np

# 定义文件路径
lc_file_path = "/home/yujie/sparse_attn/evaluation/lc_evaluation.json"
rag_file_path = "/home/yujie/sparse_attn/evaluation/rag_evaluation.json"

# 读取 Long Context 数据
with open(lc_file_path, "r") as f:
    lc_data = json.load(f)

# 读取 RAG 数据
with open(rag_file_path, "r") as f:
    rag_data = json.load(f)

# 提取指标名称和对应的值
metrics = list(lc_data.keys())  # 指标名称
lc_values = [lc_data[metric] for metric in metrics]  # Long Context 值
rag_values = [rag_data[metric] for metric in metrics]  # RAG 值

# 设置绘图参数
x = np.arange(len(metrics))  # 横坐标位置
width = 0.35  # 柱状图宽度

fig, ax = plt.subplots(figsize=(10, 6))

# 绘制柱状图
bars_lc = ax.bar(x - width / 2, lc_values, width, label="Long Context", color="skyblue")
bars_rag = ax.bar(x + width / 2, rag_values, width, label="RAG", color="orange")

# 添加标签、标题和图例
ax.set_xlabel("Metrics", fontsize=14)
ax.set_ylabel("Values", fontsize=14)
ax.set_title("Comparison of Long Context vs RAG Evaluation Metrics", fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=12)
ax.legend(fontsize=12)

# 在柱状图上显示数值
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

add_labels(bars_lc)
add_labels(bars_rag)

# 显示图表
plt.tight_layout()
plt.show()