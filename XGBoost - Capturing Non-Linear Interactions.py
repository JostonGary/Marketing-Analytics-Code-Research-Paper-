import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import shap

# Step 1: 读取潜变量得分数据
data = pd.read_csv('/Users/lirui/Desktop/4.csv')  # 确保路径正确
print(data.head())  # 检查文件是否读取正确

# Step 2: 准备数据
# 选择原始潜变量得分作为特征，'PL', 'IPS', 'ANP', 'IMM', 'ARIT' 是你的自变量
X = data[['PL', 'IPS', 'ANP', 'IMM', 'ARIT']].values

# 将 PI 分类标签作为目标变量
y = data['PI'].apply(lambda x: 1 if x > 17 else 0).values  # 假设 PI 分类为0和1

# Step 3: 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: 初始化 XGBoost 分类器
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',  # 二分类任务
    max_depth=5,  # 树的最大深度，防止过拟合
    learning_rate=0.1,  # 学习率
    n_estimators=100,  # 树的数量
    use_label_encoder=False,  # 关闭默认标签编码
    eval_metric='logloss',  # 使用logloss评估
    random_state=42
)

# Step 5: 训练模型
xgb_model.fit(X_train, y_train)

# Step 6: 进行预测
y_pred = xgb_model.predict(X_test)

# Step 7: 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred))

# Step 8: 特征重要性
importance = xgb_model.feature_importances_
feature_names = ['PL', 'IPS', 'ANP', 'IMM', 'ARIT']
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})

# Step 9: 打印和绘制特征重要性
print(importance_df)

# 可视化特征重要性
importance_df.plot(kind='barh', x='Feature', y='Importance', legend=False)
plt.title('Feature Importance in XGBoost Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()