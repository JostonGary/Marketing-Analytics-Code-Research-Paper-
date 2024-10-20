# 加载必要的库
library(tidyverse)
library(segmented)

# 读取数据
data <- read.csv("/Users/lirui/Desktop/4.csv")

# 计算XX变量，与SPSS保持一致
data$PL <- data$PL
data$PI <- data$PI

# 检查PI的范围
print(summary(data$PI))

# 选择PI范围内的一个点作为初始断点，例如中位数
initial_breakpoint <- median(data$PI)

# 进行线性回归
lm_model <- lm(PL ~ PI, data = data)

# 进行分段回归，使用计算出的初始断点
segmented_model <- segmented(lm_model, seg.Z = ~PI, psi = initial_breakpoint)

# 提取断点
breakpoint <- segmented_model$psi[2]

# 创建预测值
data$Predicted_PL <- predict(segmented_model)

# 绘图
p <- ggplot(data, aes(x = PI, y = PL, color = PI)) +
  geom_point(alpha = 0.7, size = 2) +
  geom_line(aes(y = Predicted_PL), color = "blue", size = 1) +
  geom_vline(xintercept = breakpoint, linetype = "dashed", color = "blue") +
  scale_color_gradient(low = "pink", high = "darkred") +
  labs(title = "",
       x = "Purchase Intention (PI)",
       y = "Pleasure (PL)") +
  theme_minimal() +
  theme(
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    axis.line = element_line(color = "black"),
    panel.grid = element_blank(),  # 移除所有网格线
    plot.background = element_rect(fill = "white", color = NA),  # 确保背景是白色的
    plot.margin = unit(c(1, 1, 1, 1), "cm"),
    legend.position = "none"  # 移除颜色图例
  ) +
  annotate("text", x = breakpoint, y = max(data$PL), 
           label = paste("Breakpoint =", round(breakpoint, 2)), 
           vjust = -1, hjust = -0.1, color = "blue")

# 显示图形
print(p)

# 保存图形
ggsave("PI_vs_PL_plot.png", plot = p, width = 10, height = 6, dpi = 300)

# 输出模型摘要
summary(segmented_model)

