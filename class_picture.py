import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.ticker import MultipleLocator

# 设置命令行参数
parser = argparse.ArgumentParser(
    description="Generate a bar plot from classification report JSON."
)
parser.add_argument(
    "json_file",
    type=str,
    help="Path to the JSON file containing the classification report.",
)
parser.add_argument(
    "output_image", type=str, help="Path to save the output image (e.g., output.png)."
)
args = parser.parse_args()

# 读取保存的分类报告JSON文件
with open(args.json_file, "r") as f:
    class_report = json.load(f)

# 获取类别名称和对应的precision, recall, f1-score值
labels = []
precision = []
recall = []
f1_score = []

for label, metrics in class_report.items():
    # 排除 'accuracy', 'macro avg', 'weighted avg' 等非类别项
    if label not in ["accuracy", "macro avg", "weighted avg"]:
        labels.append(label)
        precision.append(metrics["precision"])
        recall.append(metrics["recall"])
        f1_score.append(metrics["f1-score"])

# 设置柱状图的宽度
bar_width = 0.25

# 设置每个柱状图的位置
index = np.arange(len(labels))

# 使用柔和的颜色
colors = ["#66c2a5", "#fc8d62", "#8da0cb"]  # 绿色、橙色、蓝色

# 创建柱状图
plt.figure(figsize=(12, 6))
bars1 = plt.bar(index, precision, bar_width, label="Precision", color=colors[0])
bars2 = plt.bar(index + bar_width, recall, bar_width, label="Recall", color=colors[1])
bars3 = plt.bar(
    index + 2 * bar_width, f1_score, bar_width, label="F1-Score", color=colors[2]
)

# 设置x轴的标签和位置
plt.xlabel("设备类型")
plt.ylabel("分数")
# plt.title('Precision, Recall, and F1-Score for Each Device Type')
plt.xticks(index + bar_width, labels, rotation=45, ha="right")

# 设置更细粒度的y轴刻度
plt.yticks(np.arange(0, 1.2, 0.1))

# 使用 MultipleLocator 设置次刻度间隔为0.02
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.02))
# 显示主刻度线
plt.gca().yaxis.grid(
    True, which="major", linestyle=":", linewidth=0.5, color="gray"
)  # 主刻度线延伸，实线
# 显示次刻度线
plt.gca().yaxis.grid(
    True, which="minor", linestyle=":", linewidth=0.5, color="gray"
)  # 次刻度线延伸，虚线

# 在每个柱子上方标注具体数值
for bar in bars1:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 0.01,
        f"{yval:.2f}",
        ha="center",
        va="bottom",
        rotation=90,
    )

for bar in bars2:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 0.01,
        f"{yval:.2f}",
        ha="center",
        va="bottom",
        rotation=90,
    )

for bar in bars3:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 0.01,
        f"{yval:.2f}",
        ha="center",
        va="bottom",
        rotation=90,
    )


# 设置中文字体
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]  # 使用Noto Sans CJK JP字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 调整图例位置以避免遮挡内容
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

# 调整布局，增加右边空白以放置图例
plt.tight_layout(rect=[0, 0, 0.85, 1])

# 添加图例
# plt.legend()

# 调整布局
plt.tight_layout()

# 保存图片到指定路径，dpi=300
plt.savefig(args.output_image, dpi=300)

# 显示图形（可选）
plt.show()
