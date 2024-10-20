import numpy as np
import matplotlib.pyplot as plt

# 定义变量和回归系数
b_intercept = -10.2286
b_PIP = 0.8355
b_IAI = 0.9048
b_interaction = -0.0371

# 定义PIP的取值范围
PIP_values = np.linspace(-1, 1, 100)  # PIP值从-1到1（标准化后）

# 定义不同IAI值（低，中，高）
IAI_low = 23.2907
IAI_medium = 26.6054
IAI_high = 29.9201

# 计算不同IAI水平下的AY值
AY_low = b_intercept + b_PIP * PIP_values + b_IAI * IAI_low + b_interaction * PIP_values * IAI_low
AY_medium = b_intercept + b_PIP * PIP_values + b_IAI * IAI_medium + b_interaction * PIP_values * IAI_medium
AY_high = b_intercept + b_PIP * PIP_values + b_IAI * IAI_high + b_interaction * PIP_values * IAI_high

# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(PIP_values, AY_low, label='IAI (Low)', linewidth=3, color='blue')
plt.plot(PIP_values, AY_medium, label='IAI (Medium)', linewidth=3, color='green')
plt.plot(PIP_values, AY_high, label='IAI (High)', linewidth=3, color='red')
plt.xlabel('Platform Information Push (PIP)', fontsize=12, fontweight='bold')
plt.ylabel('Annoyance (AY)', fontsize=12, fontweight='bold')
plt.title('Moderating Effect of IAI on the Relationship between PIP and AY', fontsize=14, fontweight='bold')
plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.xticks(fontsize=10, fontweight='bold')
plt.yticks(fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()

















