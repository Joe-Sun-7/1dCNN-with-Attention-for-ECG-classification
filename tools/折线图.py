import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
# 数据
x = [1, 2, 3, 4, 5]
wavelet = [99.03, 99.61, 99.83, 99.74, 99.77]
fft = [98.49, 99.24, 99.52, 99.63, 99.60]
median = [98.84, 99.53, 99.76, 99.67, 99.74]

# 设置线条颜色和样式
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
linestyles = ['-', '--', '-.']

# 绘制折线图
plt.figure(figsize=(8, 6))
for i, data in enumerate([wavelet, fft, median]):
    plt.plot(x, data, color=colors[i], linestyle=linestyles[i], marker='o', markersize=5, label=['Wavelet', 'FFT', 'Median'][i])

# 添加标题和标签
plt.xlabel('Index', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Accuracy of Different Methods', fontsize=14, fontweight='bold')

# 设置刻度标签
plt.xticks(x, fontsize=10)
plt.yticks(fontsize=10)

# 添加图例
plt.legend(fontsize=10)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()
