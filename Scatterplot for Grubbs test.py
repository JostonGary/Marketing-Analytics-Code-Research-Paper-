import pandas as pd
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = '/Users/lirui/Desktop/DATA CLEAN.csv'
data = pd.read_csv(file_path)


# 使用 Grubbs 检验检测异常值
def detect_outliers_grubbs(df, column_names, alpha=0.05):
    outlier_indices = pd.Index([])

    for column_name in column_names:
        data = df[column_name]

        # 计算均值和标准差
        mean = np.mean(data)
        std = np.std(data, ddof=1)  # 样本标准差，自由度为1
        n = len(data)

        # 计算 Grubbs 检验的统计量
        G = np.abs((data - mean) / std).max()

        # 设置 Grubbs 检验的阈值
        t_critical = t.ppf(1 - alpha / (2 * n), n - 2)
        threshold = ((n - 1) / np.sqrt(n)) * np.sqrt(t_critical ** 2 / (n - 2 + t_critical ** 2))

        # 打印中间结果以便检查
        print(f"Processing column: {column_name}")
        print(f"Mean: {mean}, Std: {std}")
        print(f"Grubbs Statistic G: {G}, Threshold: {threshold}")

        # 找出 Grubbs 检验的异常值
        outliers = df[np.abs((data - mean) / std) > threshold]
        outlier_indices = outlier_indices.union(outliers.index)

    return outlier_indices


# 假设我们使用数据集中的 'AR1' 列进行异常值检测
outlier_indices = detect_outliers_grubbs(data, ['AR1'])

# 获取异常值及其对应的索引和值
outliers = data.loc[outlier_indices]

# 输出异常值的索引及其对应的值
outlier_indices_list = outlier_indices.tolist()
outlier_values_list = outliers['AR1'].tolist()

# 绘制数据的散点图，并在图中标注异常值
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['AR1'], 'b.', label='Data Points')
plt.plot(outliers.index, outliers['AR1'], 'ro', label='Outliers')

# 在异常值点上标注其索引和值
for index, value in outliers.iterrows():
    plt.text(index, value['AR1'] + 0.5, f'{index}',
             horizontalalignment='center', color='red', fontsize=10)

# 设置图表标题和轴标签
plt.title('Scatter Plot with Grubbs Outliers Highlighted', fontsize=16)
plt.xlabel('Index', fontsize=14)
plt.ylabel('AR1 Value', fontsize=14)
plt.grid(linestyle='--', alpha=0.7)
plt.legend()
plt.show()

outlier_indices_list, outlier_values_list
