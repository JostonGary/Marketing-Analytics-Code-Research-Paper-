import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

# 提高图形分辨率
plt.rcParams['figure.dpi'] = 100  # 调整图形显示分辨率

# Step 1: 读取数据
data = pd.read_csv('/Users/lirui/Desktop/4.csv')  # 请根据实际路径修改
X = data[['PL', 'IPS', 'ANP', 'IMM', 'ARIT']].values  # 自变量
y = data['PI'].apply(lambda x: 1 if x > 17 else 0).values  # 因变量，PI的分类

# Step 2: 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: 定义 XGBoost 模型
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Step 4: 训练模型
model.fit(X_train, y_train)

# Step 5: 预测并评估模型
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred))

# Step 6: AUC-ROC
y_pred_proba = model.predict_proba(X_test)[:, 1]  # 获取概率预测值
auc = roc_auc_score(y_test, y_pred_proba)
print(f'AUC-ROC: {auc:.4f}')

# Step 7: 绘制并保存 ROC 曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))  # 调整窗口大小
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('/Users/lirui/Desktop/roc_curve.png')  # 保存图像
plt.show()

# Step 8: 使用 SHAP 解释模型
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Step 9: 可视化并保存 SHAP summary plot
plt.figure(figsize=(8, 6))  # 调整窗口大小
shap.summary_plot(shap_values, X_test, feature_names=['PL', 'IPS', 'ANP', 'IMM', 'ARIT'])
plt.savefig('/Users/lirui/Desktop/shap_summary.png')  # 保存图像
plt.show()

# Step 10: SHAP 决策图并保存
plt.figure(figsize=(8, 6))  # 调整窗口大小
shap.decision_plot(explainer.expected_value, shap_values[0], feature_names=['PL', 'IPS', 'ANP', 'IMM', 'ARIT'])
plt.savefig('/Users/lirui/Desktop/shap_decision_plot.png')  # 保存图像
plt.show()

# Step 11: SHAP 特征重要性条形图并保存
plt.figure(figsize=(8, 6))  # 调整窗口大小
shap.bar_plot(shap_values, feature_names=['PL', 'IPS', 'ANP', 'IMM', 'ARIT'])
plt.savefig('/Users/lirui/Desktop/shap_bar_plot.png')  # 保存图像
plt.show()

# Step 12: 部分依赖图（PDP）并保存
fig, ax = plt.subplots(figsize=(8, 6))  # 调整窗口大小
# 检查 features 参数，确保索引正确
PartialDependenceDisplay.from_estimator(model, X_test, features=[0, 1], feature_names=['PL', 'IPS', 'ANP', 'IMM', 'ARIT'], ax=ax)
plt.savefig('/Users/lirui/Desktop/pdp_plot.png')  # 保存图像
plt.show()

# Step 13: 打印预测结果
print("预测结果:", y_pred)

# Step 14: 可视化决策树结构并保存
fig, ax = plt.subplots(figsize=(20, 10))  # 调整窗口大小，确保足够大显示树结构
xgb.plot_tree(model, num_trees=0, ax=ax)  # 只显示第1棵树，你可以根据需求更改 num_trees
plt.savefig('/Users/lirui/Desktop/tree_structure.png')  # 保存图像
plt.show()


print("所有图表已显示并保存到指定路径。")