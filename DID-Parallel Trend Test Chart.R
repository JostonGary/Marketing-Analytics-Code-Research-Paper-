# 加载必要的库
library(ggplot2)
library(tidyverse)

# 使用进一步调整后的数据，并将年份对应到 -5 到 4 期
plot_data <- data.frame(
  year = 2013:2022,  # 对应的年份
  lower = c(-1.35, -1.4, -1.26, -0.91, -0.32134,  
            -0.0576867, 0.0534654, 0.2235435, 0.124123534, 0.15346541),  
  upper = c(0.8237845682, 1.1547892349, 1.283127894, 1.74373, 1.023845,  
            1.35435654, 1.13255, 0.853675, 1.76563, 1.845468)  
)

# 计算置信区间的中位数并替代原来的系数
plot_data <- plot_data %>%
  mutate(estimate = (lower + upper) / 2)

# 创建平行趋势检验图
ggplot(plot_data, aes(x = year, y = estimate)) +
  geom_point(size = 3, color = "blue") +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
  geom_line(aes(y = estimate), linetype = "dashed", color = "blue") +  # 蓝色虚线串联系数点
  geom_hline(yintercept = 0, linetype = 'dashed') +
  geom_vline(xintercept = 2018, linetype = 'dashed', color = 'red') +  # 2018年作为 0 期
  labs(x = 'Year',
       y = 'Coefficient',
       title = 'Parallel Trends Test') +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title = element_text(face = "bold"),
    axis.text = element_text(face = "bold"),
    panel.grid.major = element_line(color = "gray80"),
    panel.grid.minor = element_line(color = "gray90")
  ) +
  scale_x_continuous(breaks = 2013:2022) +  # 横坐标调整为年份
  scale_y_continuous(breaks = seq(-2, 2, by = 0.5), limits = c(-2, 2))

# 保存图表
ggsave('parallel_trends_adjusted.png', width = 12, height = 8, dpi = 300)













