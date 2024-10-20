library(ggplot2)

# 定义变量和回归系数
b_intercept <- -10.2286
b_PIP <- 0.8355
b_IAI <- 0.9048
b_interaction <- -0.0371

# 定义PIP的取值范围
PIP_values <- seq(-1, 1, by = 0.1)  # PIP值从-1到1（标准化后）

# 定义不同IAI值的范围
IAI_values <- seq(15, 35, by = 1)  # IAI值从15到35

# 计算不同IAI水平下的AY值
results <- data.frame()
for (IAI in IAI_values) {
  AY_values <- b_intercept + b_PIP * PIP_values + b_IAI * IAI + b_interaction * PIP_values * IAI
  temp_df <- data.frame(PIP = PIP_values, AY = AY_values, IAI = IAI)
  results <- rbind(results, temp_df)
}

# 绘制敏感性分析图表
ggplot(results, aes(x = PIP, y = AY, color = as.factor(IAI))) +
  geom_line(size = 1) +
  labs(
    title = "Sensitivity Analysis of IAI on the Relationship between PIP and AY",
    x = "Platform Information Push (PIP)",
    y = "Annoyance (AY)",
    color = "IAI Levels"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),   # 标题居中并加粗
    axis.title = element_text(face = "bold"),                # 坐标轴标题加粗
    axis.text = element_text(face = "bold")                  # 坐标轴字体加粗
  )

