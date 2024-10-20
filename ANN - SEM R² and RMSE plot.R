# 加载必要的库
library(ggplot2)
library(gridExtra)

# 假设你有以下数据
train_rmses <- c(0.223, 0.235, 0.231, 0.227, 0.230, 0.227, 0.229, 0.224, 0.231, 0.229)
test_rmses <- c(0.239, 0.221, 0.237, 0.243, 0.223, 0.235, 0.230, 0.240, 0.227, 0.242)
train_r2s <- c(0.651, 0.632, 0.638, 0.644, 0.641, 0.644, 0.640, 0.650, 0.637, 0.639)
test_r2s <- c(0.615, 0.625, 0.618, 0.611, 0.623, 0.619, 0.622, 0.614, 0.626, 0.612)

# 数据框架
data <- data.frame(
  Iteration = rep(1:10, 2),
  RMSE = c(train_rmses, test_rmses),
  R2 = c(train_r2s, test_r2s),
  Type = rep(c("Training", "Testing"), each = 10)
)

# RMSE箱线图
p1 <- ggplot(data, aes(x = Type, y = RMSE)) +
  geom_boxplot() +
  labs(title = "Boxplot of RMSE") +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.text = element_text(size = 12, face = "bold"),
    axis.title = element_text(size = 12, face = "bold")
  )

# RMSE折线图
p2 <- ggplot(data, aes(x = Iteration, y = RMSE, color = Type)) +
  geom_line() +
  geom_point() +
  labs(title = "Line plot of RMSE", x = "Iteration", y = "RMSE") +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.text = element_text(size = 12, face = "bold"),
    axis.title = element_text(size = 12, face = "bold"),
    legend.title = element_blank()
  )

# R²箱线图
p3 <- ggplot(data, aes(x = Type, y = R2)) +
  geom_boxplot() +
  labs(title = "Boxplot of R²") +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.text = element_text(size = 12, face = "bold"),
    axis.title = element_text(size = 12, face = "bold")
  )

# R²折线图
p4 <- ggplot(data, aes(x = Iteration, y = R2, color = Type)) +
  geom_line() +
  geom_point() +
  labs(title = "Line plot of R²", x = "Iteration", y = "R²") +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.text = element_text(size = 12, face = "bold"),
    axis.title = element_text(size = 12, face = "bold"),
    legend.title = element_blank()
  )

# 合并所有图表
grid.arrange(p1, p2, p3, p4, ncol = 2)
