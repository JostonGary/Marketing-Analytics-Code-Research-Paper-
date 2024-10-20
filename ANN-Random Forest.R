install.packages("caret")
install.packages("randomForest")

# 加载必要的库
library(caret)
library(randomForest)

# 设置文件路径
file_path <- "/Users/lirui/Downloads/Processed_Thesis_data.csv"

# 加载数据集
data_corrected <- read.csv(file_path)

# 检查数据集的前几行
head(data_corrected)

# 定义输入和输出变量
input_vars <- c("PIP", "PL", "AY", "IAI", "PI")
output_var <- "PB"

X <- data_corrected[, input_vars]
y <- data_corrected[, output_var]

# 定义5折交叉验证
control <- trainControl(method = "cv", number = 5)

# 调整随机森林模型的参数
tune_grid <- expand.grid(mtry = c(2, 3, 4))

# 重新训练模型
model_tuned <- train(X, y, method = "rf", trControl = control, tuneGrid = tune_grid)

# 打印调整后的模型结果
print(model_tuned)

# 获取预测结果
predictions_tuned <- predict(model_tuned, newdata = X)
results_tuned <- data.frame(Observed = y, Predicted = predictions_tuned)

# 计算 RMSE 和 R²
rmse_tuned <- sqrt(mean((results_tuned$Observed - results_tuned$Predicted)^2))
r2_tuned <- cor(results_tuned$Observed, results_tuned$Predicted)^2

# 打印结果
print(paste("Tuned RMSE:", rmse_tuned))
print(paste("Tuned R²:", r2_tuned))

(分割线-----------------------------------------)


# 使用Boruta包进行特征选择
install.packages("Boruta")
library(Boruta)

# 进行特征选择
boruta_output <- Boruta(X, y, doTrace = 2)

# 打印重要特征
print(boruta_output)

# 获取最终确认的特征
final_features <- getSelectedAttributes(boruta_output, withTentative = F)

# 打印最终确认的特征
print(final_features)


(分割线-----------------------------------------)


# 安装并加载ggplot2包
install.packages("ggplot2")
library(ggplot2)

# 绘制实际值与预测值的散点图
ggplot(results, aes(x = Observed, y = Predicted)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = 'red') +
  labs(title = "Actual vs Predicted", x = "Actual Values", y = "Predicted Values") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    axis.title.x = element_text(size = 12, face = "bold"),
    axis.title.y = element_text(size = 12, face = "bold"),
    axis.text.x = element_text(size = 10, face = "bold"),
    axis.text.y = element_text(size = 10, face = "bold")
  )


(分割线-----------------------------------------)


# 加载必要的库
library(ggplot2)

# 绘制残差图
residuals <- results$Observed - results$Predicted
ggplot(data.frame(Residuals = residuals), aes(x = Residuals)) +
  geom_histogram(binwidth = 0.05, fill = 'blue', color = 'black', alpha = 0.7) +
  labs(title = "Residuals Histogram", x = "Residuals", y = "Frequency") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    axis.title.x = element_text(size = 12, face = "bold"),
    axis.title.y = element_text(size = 12, face = "bold"),
    axis.text.x = element_text(size = 10, face = "bold"),
    axis.text.y = element_text(size = 10, face = "bold")
  )



