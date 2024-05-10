import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'

hyperparameters = [
    "d=0.2, lr=0.001",
    "d=0.2, lr=0.003",
    "d=0.2, lr=0.005",
    "d=0.3, lr=0.001",
    "d=0.3, lr=0.003",
    "d=0.3, lr=0.005",
    "d=0.1, lr=0.003",
    "d=0.3, lr=0.01"
]
accuracy = [
    99.13, 99.55, 99.34, 99.37, 99.60, 99.34, 99.31, 97.81
]

# 创建颜色列表
colors = ['#f4f1de', '#df7a5e', '#3c405b', '#82b29a', '#f2cc8e']

# 创建直方图
plt.figure(figsize=(10, 6))
bars = plt.bar(hyperparameters, accuracy, color=colors, edgecolor='black')

# 设置纵轴范围从99%开始
plt.ylim(95, 100)

# 在每个柱子上方标出数据
for bar, acc in zip(bars, accuracy):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f'{acc:.2f}', ha='center', va='bottom', fontsize=9)

# 添加标题和标签
plt.xlabel('Hyperparameter Settings', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy(%)', fontsize=12, fontweight='bold')
plt.title('Hyperparameter Settings vs Accuracy', fontsize=14, fontweight='bold')

# 旋转x轴标签
plt.xticks(rotation=45, ha='right')

# 显示图形
plt.tight_layout()
plt.show()
