import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

# 1. 数据预处理
# 假设您的数据已经被保存为CSV文件
data = pd.read_csv('/Users/lirui/Downloads/Processed_Thesis_data.csv')

# 删除IAI变量
data = data.drop('IAI', axis=1)

# 分离特征和目标变量
X = data[['PIP', 'PL', 'AY', 'PI']]
y = data['PB']

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2. 构建神经网络模型
model = Sequential([
    Dense(10, activation='relu', input_shape=(4,), name='hidden_layer_1'),
    Dense(5, activation='relu', name='hidden_layer_2'),
    Dense(1, activation='linear', name='output_layer')
])

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 3. 训练模型
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# 4. 提取权重、偏置和激活值
def get_layer_output(model, layer_name, input_data):
    intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                              outputs=model.get_layer(layer_name).output)
    return intermediate_layer_model.predict(input_data)

# 获取权重和偏置
for layer in model.layers:
    weights, biases = layer.get_weights()
    print(f"\n{layer.name} Weights:")
    print(weights)
    print(f"\n{layer.name} Biases:")
    print(biases)

# 获取各层的激活值（使用训练集的一个小批次作为示例）
sample_batch = X_train[:32]
for layer in model.layers:
    activations = get_layer_output(model, layer.name, sample_batch)
    print(f"\n{layer.name} Activations (first 5 samples, first 5 units):")
    print(activations[:5, :5])

# 打印模型结构
model.summary()

# 评估模型
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss}")

# 使用模型进行预测
predictions = model.predict(X_test)
print("\nSample Predictions (first 5):")
print(predictions[:5])
















