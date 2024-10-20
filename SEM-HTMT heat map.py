import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 定义文件路径
file_path = '/Users/lirui/Desktop/4.csv'

# 加载CSV文件到DataFrame
data = pd.read_csv(file_path)

# 提取已合并的潜变量列（确保这些列名与CSV文件中的列名一致）
latent_variables = ["AR", "IT", "IVR", "PSN", "IMM", "ARIT", "PL", "IPS", "ANP", "PI", "CBI"]

def calculate_htmt(data, latent_variables):
    # 创建一个空的HTMT矩阵
    htmt_matrix = pd.DataFrame(np.zeros((len(latent_variables), len(latent_variables))),
                               index=latent_variables, columns=latent_variables)

    for i, lv1 in enumerate(latent_variables):
        for j, lv2 in enumerate(latent_variables):
            if i >= j:
                continue

            # 计算潜变量之间的相关性
            correlation = data[lv1].corr(data[lv2])
            htmt_matrix.iloc[i, j] = correlation
            htmt_matrix.iloc[j, i] = correlation

    return htmt_matrix

# 将所有列转换为数值型，以防止非数值数据导致的计算错误
data_numeric = data[latent_variables].apply(pd.to_numeric, errors='coerce')

# 计算HTMT
htmt_matrix = calculate_htmt(data_numeric, latent_variables)

# 设置 Pandas 的显示选项，确保完整显示 HTMT 矩阵
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)     # 显示所有行
pd.set_option('display.width', 1000)        # 设置显示的宽度，以避免自动换行

# 输出 HTMT 矩阵
print("HTMT Matrix:")
print(htmt_matrix)

# 如果需要，将 HTMT 矩阵导出为 CSV 文件
htmt_matrix.to_csv('/Users/lirui/Desktop/output_htmt_matrix.csv', index=True)
print("HTMT 矩阵已导出为 CSV 文件")

# 使用热图可视化 HTMT 矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(htmt_matrix, annot=True, fmt=".3f", cmap="coolwarm")
plt.title("HTMT Matrix Heatmap")
plt.show()
