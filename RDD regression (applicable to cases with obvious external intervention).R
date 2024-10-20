# 安装并加载必要的包
if (!require(rdrobust)) install.packages("rdrobust")
library(rdrobust)

# 读取数据
data <- read.csv("/Users/lirui/Desktop/4.csv")

# 设置运行变量（X）和结果变量（Y）
X <- data$PI  # 假设PI是运行变量
Y <- data$PL  # 假设PL是结果变量

# 计算截断点（这里使用中位数，您可能需要根据实际情况调整）
cutoff <- median(X)

# 使用rdrobust函数进行RDD分析，指定截断点
rdd_result <- rdrobust(Y, X, c = cutoff)

# 使用rdplot函数绘图
rdplot(Y, X, c = cutoff,
       title = "Purchase Intention vs Pleasure",
       x.label = "Purchase Intention (PI)",
       y.label = "Pleasure (PL)")

# 保存图形
dev.copy(png, "RDD_PI_vs_PL_plot.png", width = 800, height = 600)
dev.off()

# 打印RDD分析结果摘要
summary(rdd_result)

