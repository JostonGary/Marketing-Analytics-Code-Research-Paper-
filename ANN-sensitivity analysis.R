install.packages("caret")
install.packages("nnet")  # 用于神经网络模型
install.packages("NeuralNetTools")  # 用于计算特征重要性
install.packages("caret")  # 用于模型训练和交叉验证

# 加载必要的库
library(caret)
library(nnet)
library(NeuralNetTools)

# 设置文件路径
file_path <- "/Users/lirui/Desktop/硕博/Manuscript/Manuscript-1\ 原始数据/Processed_Thesis_data.csv"

# 加载数据集
data_corrected <- read.csv(file_path)

# 定义输入和输出变量
input_vars <- c("PIP", "PL", "AY", "IAI", "PI")
output_var <- "PB"

X <- data_corrected[, input_vars]
y <- data_corrected[, output_var]

(分割线--------------------10次迭代结果合并到1次---------------------)

# 初始化存储特征重要性的矩阵
importance_matrix <- matrix(0, nrow = 10, ncol = length(input_vars))
colnames(importance_matrix) <- input_vars

# 进行10次迭代
for (i in 1:10) {
  # 定义ANN模型
  model <- nnet(X, y, size = 5, linout = TRUE, trace = FALSE)
  
  # 计算每个特征的重要性，使用Garson's算法
  var_imp <- garson(model, bar_plot = FALSE)$rel_imp
  
  # 存储特征重要性
  importance_matrix[i, ] <- var_imp
}

# 计算平均值和标准化的重要性
average_importance <- colMeans(importance_matrix)
normalized_importance <- average_importance / max(average_importance) * 100

# 创建数据框以显示结果
results <- data.frame(
  Feature = names(average_importance),
  Average = average_importance,
  NormalizedImportance = normalized_importance
)

# 打印结果
print(results)

(分割线---------------------如果需要每一迭代的结果-------------------)

# 安装所需的包（如果尚未安装）
install.packages(c("caret", "nnet", "NeuralNetTools"))

# 加载必要的库
library(caret)
library(nnet)
library(NeuralNetTools)

# 设置文件路径（根据需要调整）
file_path <- "/Users/lirui/Desktop/硕博/Manuscript/Manuscript-1\ 原始数据/Processed_Thesis_data.csv"

# 加载数据集
data_corrected <- read.csv(file_path)

# 定义输入和输出变量
input_vars <- c("PIP", "PL", "AY", "IAI", "PI")
output_var <- "PB"

X <- data_corrected[, input_vars]
y <- data_corrected[, output_var]

# 初始化矩阵以存储每次迭代的特征重要性
importance_matrix <- matrix(0, nrow = 10, ncol = length(input_vars))
colnames(importance_matrix) <- input_vars

# 进行10次迭代
for (i in 1:10) {
  # 定义ANN模型
  model <- nnet(X, y, size = 5, linout = TRUE, trace = FALSE)
  
  # 使用Garson算法计算特征重要性
  var_imp <- garson(model, bar_plot = FALSE)$rel_imp
  
  # 存储特征重要性
  importance_matrix[i, ] <- var_imp
  
  # 打印每次迭代的结果
  cat("第", i, "次迭代:\n")
  print(var_imp)
  cat("\n")
}

# 计算平均和标准化的重要性
average_importance <- colMeans(importance_matrix)
normalized_importance <- average_importance / max(average_importance) * 100

# 创建包含结果的数据框
results <- data.frame(
  特征 = names(average_importance),
  平均值 = average_importance,
  标准化重要性 = normalized_importance
)

# 打印总体结果
cat("\n总体结果:\n")
print(results)

# 打印完整的重要性矩阵
cat("\n完整的重要性矩阵:\n")
print(importance_matrix)

