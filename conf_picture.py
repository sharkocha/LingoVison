import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse

# 设置命令行参数
parser = argparse.ArgumentParser(
    description="Generate a normalized confusion matrix heatmap."
)
parser.add_argument(
    "csv_file", type=str, help="Path to the CSV file containing the confusion matrix."
)
parser.add_argument(
    "output_image", type=str, help="Path to save the output heatmap image."
)
args = parser.parse_args()

# 读取CSV文件中的混淆矩阵
conf_matrix_df = pd.read_csv(args.csv_file, index_col=0)

# 将数据框转换为numpy数组
conf_matrix = conf_matrix_df.values

# 对混淆矩阵进行按行归一化（即每一行的总和为1）
conf_matrix_normalized = (
    conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
)

# 获取类别标签（行/列名）
device_types = conf_matrix_df.columns

# 生成归一化后的热力图
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=list(device_types),
    yticklabels=list(device_types),
)

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]  # 使用Noto Sans CJK JP字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 设置图表标题和坐标轴标签
# plt.title('Normalized Confusion Matrix Heatmap')
plt.xlabel("真实标签")
plt.ylabel("预测标签")

# 调整刻度标签的角度
plt.xticks(rotation=45, ha="right")  # 标签旋转45度，ha='right'将对齐方式设置为右对齐
plt.yticks(rotation=0)  # 纵轴标签保持水平

# 调整布局以防止标签被截断
plt.tight_layout()

# 保存图片，dpi=300
plt.savefig(args.output_image, dpi=300)

# 显示热力图（可选，生成图片时可以不显示）
plt.show()
