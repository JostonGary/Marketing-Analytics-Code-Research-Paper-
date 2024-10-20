import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# 读取数据
file_path = '/Users/lirui/Desktop/DATA CLEAN.csv'  # 替换为你的文件路径
data = pd.read_csv(file_path)

# 计算构念之间的相关矩阵
correlation_matrix = data[['AR', 'IT', 'IV', 'PS', 'IM', 'ARTR', 'PL', 'IS', 'AN', 'PI', 'CPI']].corr()

# Fornell-Larcker标准
ave_values = {
    'AR': data['AR'].var(),
    'IT': data['IT'].var(),
    'IV': data['IV'].var(),
    'PS': data['PS'].var(),
    'IM': data['IM'].var(),
    'ARTR': data['ARTR'].var(),
    'PL': data['PL'].var(),
    'IS': data['IS'].var(),
    'AN': data['AN'].var(),
    'PI': data['PI'].var(),
    'CPI': data['CPI'].var()
}

ave_sqrt = {key: np.sqrt(value) for key, value in ave_values.items()}

print("Fornell-Larcker Criterion")
for key in ave_sqrt:
    print(f"{key}: AVE sqrt = {ave_sqrt[key]:.2f}")

# HTMT计算
def htmt(correlation_matrix):
    htmt_values = {}
    for i in range(correlation_matrix.shape[0]):
        for j in range(i + 1, correlation_matrix.shape[0]):
            trait_corr = correlation_matrix.iloc[i, j]
            htmt_values[(correlation_matrix.index[i], correlation_matrix.columns[j])] = trait_corr
    return htmt_values

htmt_values = htmt(correlation_matrix)

print("HTMT Values")
for key, value in htmt_values.items():
    print(f"{key}: {value}")