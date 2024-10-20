import pandas as pd
import numpy as np

# 读取CSV文件
file_path = '/Users/lirui/Desktop/DATA CLEAN.csv'  # 替换为你的文件路径
data = pd.read_csv(file_path)


# 计算Cronbach's Alpha
def cronbach_alpha(df):
    item_count = df.shape[1]
    item_variances = df.var(axis=0, ddof=1)
    total_score_variance = df.sum(axis=1).var(ddof=1)
    alpha = (item_count / (item_count - 1)) * (1 - item_variances.sum() / total_score_variance)
    return alpha


# 计算Composite Reliability (CR) 和 Average Variance Extracted (AVE)
def calculate_cr_ave(df):
    # 假设因子载荷可以用项目与总分的相关系数估计
    total_score = df.sum(axis=1)
    loadings = df.apply(lambda x: x.corr(total_score))

    # 计算CR
    sum_lambda = loadings.sum()
    sum_lambda_squared = (loadings ** 2).sum()
    cr = sum_lambda ** 2 / (sum_lambda ** 2 + (len(loadings) - sum_lambda_squared))

    # 计算AVE
    ave = sum_lambda_squared / len(loadings)

    return cr, ave


# 定义你要分析的构念
constructs = {
    'AR': ['AR1', 'AR2', 'AR3', 'AR4', 'AR5', 'AR6'],
    'IT': ['IT1', 'IT2', 'IT3', 'IT4', 'IT5', 'IT6'],
    'IV': ['IV1', 'IV2', 'IV3', 'IV4', 'IV5'],
    'PS': ['PS1', 'PS2', 'PS3'],
    'IM': ['IM1', 'IM2', 'IM3'],
    'ARTR': ['ARTR1', 'ARTR2', 'ARTR3'],
    'PL': ['PL1', 'PL2', 'PL3'],
    'IS': ['IS1', 'IS2', 'IS3', 'IS4', 'IS5'],
    'AN': ['AN1', 'AN2', 'AN3', 'AN4', 'AN5', 'AN6'],
    'PI': ['PI1', 'PI2', 'PI3', 'PI4', 'PI5'],
    'CPI': ['CPI1', 'CPI2', 'CPI3', 'CPI4', 'CPI5']
}

# 计算并输出每个构念的信度指标
for construct_name, items in constructs.items():
    df_construct = data[items].dropna()  # 获取每个构念的数据
    alpha = cronbach_alpha(df_construct)
    cr, ave = calculate_cr_ave(df_construct)
    print(f"{construct_name} - Cronbach's Alpha: {alpha:.3f}, CR: {cr:.3f}, AVE: {ave:.3f}")



