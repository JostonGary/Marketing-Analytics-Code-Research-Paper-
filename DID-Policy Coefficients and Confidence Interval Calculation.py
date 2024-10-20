import pandas as pd
import statsmodels.api as sm

# 创建数据框
data = {
    'Company': ['BYD', 'BYD', 'BYD', 'BYD', 'BYD', 'BYD', 'BYD', 'BYD', 'BYD', 'BYD',
                'CHANGAN', 'CHANGAN', 'CHANGAN', 'CHANGAN', 'CHANGAN', 'CHANGAN', 'CHANGAN', 'CHANGAN', 'CHANGAN', 'CHANGAN',
                'CHANGCHENG', 'CHANGCHENG', 'CHANGCHENG', 'CHANGCHENG', 'CHANGCHENG', 'CHANGCHENG', 'CHANGCHENG', 'CHANGCHENG', 'CHANGCHENG', 'CHANGCHENG',
                'GUANGQI', 'GUANGQI', 'GUANGQI', 'GUANGQI', 'GUANGQI', 'GUANGQI', 'GUANGQI', 'GUANGQI', 'GUANGQI', 'GUANGQI',
                'SHANGQI', 'SHANGQI', 'SHANGQI', 'SHANGQI', 'SHANGQI', 'SHANGQI', 'SHANGQI', 'SHANGQI', 'SHANGQI', 'SHANGQI',
                'YUTONG', 'YUTONG', 'YUTONG', 'YUTONG', 'YUTONG', 'YUTONG', 'YUTONG', 'YUTONG', 'YUTONG', 'YUTONG'],
    'Year': [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022] * 6,
    'Policy': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 6,
    'Treat': [1] * 30 + [0] * 30,
    'PatentCount': [418, 804, 947, 1577, 1880, 2224, 2095, 1930, 1483, 2169,
                    169, 330, 531, 821, 911, 915, 661, 894, 1199, 818,
                    251, 428, 415, 863, 783, 970, 989, 1271, 1965, 2345,
                    5, 16, 62, 38, 135, 255, 662, 586, 902, 1290,
                    49, 166, 429, 659, 1116, 1502, 1435, 1436, 1468, 992,
                    33, 74, 136, 202, 575, 446, 198, 171, 70, 105]
}

df = pd.DataFrame(data)

# 创建相对年份
df['relative_year'] = df['Year'] - 2018

# 创建交互项
for year in range(-5, 5):
    df[f'interact_{year}'] = (df['relative_year'] == year) * df['Treat']

# 运行回归
X = df[[f'interact_{year}' for year in range(-5, 5)] + ['Treat']]
X = sm.add_constant(X)
y = df['PatentCount']

model = sm.OLS(y, X).fit()

# 提取系数和置信区间
coef = model.params[1:10]
conf_int = model.conf_int().iloc[1:10]

# 打印系数和置信区间
print("Coefficients:\n", coef)
print("\nConfidence Intervals:\n", conf_int)

import numpy as np

# 计算缩放因子
max_abs_coef = np.max(np.abs(coef))
scaling_factor = 1.0 / max_abs_coef

# 缩放系数
scaled_coef = coef * scaling_factor
scaled_conf_int = conf_int * scaling_factor

# 打印缩放后的系数和置信区间
print("Scaled Coefficients:\n", scaled_coef)
print("\nScaled Confidence Intervals:\n", scaled_conf_int)