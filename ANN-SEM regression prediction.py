import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns


# 1. 数据预处理
data = pd.read_csv('/Users/lirui/Downloads/Processed_Thesis_data.csv')
data = data.drop('IAI', axis=1)

X = data[['PIP', 'PL', 'AY', 'PI']]
y = data['PB']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2. 构建改进后的神经网络模型
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],), name='hidden_layer_1'),
    Dense(32, activation='relu', name='hidden_layer_2'),
    Dense(16, activation='relu', name='hidden_layer_3'),
    Dense(1, activation='linear', name='output_layer')
])

# 改用 Huber loss，减少大残差的影响
model.compile(optimizer='adam', loss=tf.keras.losses.Huber())

# 训练模型
history = model.fit(X_train, y_train, epochs=150, batch_size=16, validation_split=0.2, verbose=1)

# 3. 进行预测
predictions = model.predict(X_test).flatten()

# 计算误差
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)

print(f"\nMean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")

# 可视化实际值与预测值的对比
plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=predictions, ci=None, scatter_kws={'s': 40}, line_kws={'color': 'red'})
plt.title("Actual vs Predicted Values with Regression Line")
plt.xlabel("Actual PB Values")
plt.ylabel("Predicted PB Values")
plt.show()

# 4. 打印实际值和预测值的对比
for i in range(5):
    print(f"Actual: {y_test.iloc[i]}, Predicted: {predictions[i]}")

# 可视化实际值与预测值的对比
plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:50], label="Actual Values")
plt.plot(predictions[:50], label="Predicted Values", linestyle='dashed')
plt.title("Actual vs Predicted Values (First 50 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("PB Value")
plt.legend()
plt.show()


