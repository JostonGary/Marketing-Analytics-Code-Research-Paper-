import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_recall_fscore_support, roc_curve
import matplotlib.pyplot as plt

# 读取潜变量得分数据
data = pd.read_csv('/Users/lirui/Desktop/4.csv')
latent_scores = data[['PL', 'IPS', 'ANP', 'IMM', 'ARIT']].values

# 计算余弦相似度和邻接矩阵
similarity_matrix = cosine_similarity(latent_scores)
threshold = 0.85
adjacency_matrix = (similarity_matrix > threshold).astype(int)

# 使用邻接矩阵创建图
G = nx.from_numpy_array(adjacency_matrix)
for i, node in enumerate(G.nodes):
    G.nodes[node]['feature'] = latent_scores[i]

# 将 NetworkX 图转换为 PyTorch Geometric 格式
graph_data = from_networkx(G)
graph_data.x = torch.tensor(latent_scores, dtype=torch.float)

# 将 PI 分数转化为分类标签
data['PI_category'] = data['PI'].apply(lambda pi_value: 0 if pi_value <= 17 else 1)
labels = torch.tensor(data['PI_category'].values)

# 使用 train_test_split 划分训练集和验证集索引
train_indices, val_indices = train_test_split(range(graph_data.num_nodes), test_size=0.2, random_state=42)

# 转换为 PyTorch 张量
train_indices = torch.tensor(train_indices, dtype=torch.long)
val_indices = torch.tensor(val_indices, dtype=torch.long)

# 定义情绪变量和认知变量的索引
affective_indices = [0, 1, 2]  # 对应'PL', 'IPS', 'ANP'
cognitive_indices = [3, 4]  # 对应'IMM', 'ARIT'

# 定义注意力层（分别处理情绪变量和认知变量）
class AffectiveAttentionLayer(nn.Module):
    def __init__(self, affective_dim):
        super(AffectiveAttentionLayer, self).__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(affective_dim, 1))
        nn.init.xavier_uniform_(self.attention_weights)

    def forward(self, x):
        attention_scores = torch.matmul(x, self.attention_weights)
        attention_scores = F.softmax(attention_scores, dim=1)
        weighted_x = x * attention_scores
        return weighted_x

class CognitiveAttentionLayer(nn.Module):
    def __init__(self, cognitive_dim):
        super(CognitiveAttentionLayer, self).__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(cognitive_dim, 1))
        nn.init.xavier_uniform_(self.attention_weights)

    def forward(self, x):
        attention_scores = torch.matmul(x, self.attention_weights)
        attention_scores = F.softmax(attention_scores, dim=1)
        weighted_x = x * attention_scores
        return weighted_x

# 定义应用注意力机制的GCN模型
class AttentionGCN(nn.Module):
    def __init__(self, affective_dim, cognitive_dim, hidden_dim, num_classes):
        super(AttentionGCN, self).__init__()
        self.affective_attention = AffectiveAttentionLayer(affective_dim)  # 情绪注意力层
        self.cognitive_attention = CognitiveAttentionLayer(cognitive_dim)  # 认知注意力层
        self.conv1 = GCNConv(affective_dim + cognitive_dim, hidden_dim)  # 连接情绪和认知变量后的卷积层
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        affective_x = x[:, affective_indices]  # 提取情绪变量特征
        cognitive_x = x[:, cognitive_indices]  # 提取认知变量特征

        # 分别通过情绪和认知的注意力层
        affective_x = self.affective_attention(affective_x)
        cognitive_x = self.cognitive_attention(cognitive_x)

        # 合并情绪和认知特征
        x = torch.cat([affective_x, cognitive_x], dim=1)

        # 通过GCN层
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 假设我们已经有了图数据
num_affective_features = len(affective_indices)  # 情绪变量的数量
num_cognitive_features = len(cognitive_indices)  # 认知变量的数量
num_classes = 2  # 二分类任务

# 实例化模型
model = AttentionGCN(num_affective_features, num_cognitive_features, hidden_dim=16, num_classes=num_classes)

# 损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
train_loss_values = []
val_loss_values = []
train_acc_values = []
val_acc_values = []

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(graph_data)
    loss = criterion(out[train_indices], labels[train_indices])
    loss.backward()
    optimizer.step()

    # 计算训练集准确率
    pred_train = out[train_indices].max(1)[1]
    correct_train = pred_train.eq(labels[train_indices]).sum().item()
    train_acc = correct_train / len(train_indices)
    train_loss_values.append(loss.item())
    train_acc_values.append(train_acc)

    # 验证集
    model.eval()
    with torch.no_grad():
        val_out = model(graph_data)
        val_loss = criterion(val_out[val_indices], labels[val_indices])
        val_loss_values.append(val_loss.item())
        pred_val = val_out[val_indices].max(1)[1]
        correct_val = pred_val.eq(labels[val_indices]).sum().item()
        val_acc = correct_val / len(val_indices)
        val_acc_values.append(val_acc)

    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Train Loss: {loss.item()}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss.item()}, Val Acc: {val_acc:.4f}')

# 输出分类报告
precision, recall, f1_score, support = precision_recall_fscore_support(labels[val_indices].numpy(), pred_val.numpy())
accuracy = accuracy_score(labels[val_indices].numpy(), pred_val.numpy())
auc_roc = roc_auc_score(labels[val_indices].numpy(), F.softmax(val_out[val_indices], dim=1)[:, 1].numpy())

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1_score)
print("Support:", support)
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")

# 可视化结果
# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(200), train_loss_values, label='Train Loss', color='blue')
plt.plot(range(200), val_loss_values, label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss with Attention Mechanism')
plt.legend()
plt.show()


# 绘制 ROC 曲线
fpr, tpr, _ = roc_curve(labels[val_indices].numpy(), F.softmax(val_out[val_indices], dim=1)[:, 1].numpy())
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_roc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

#绘制PRC曲线
from sklearn.metrics import precision_recall_curve, auc

probs = F.softmax(val_out[val_indices], dim=1)[:, 1].detach().numpy()  # 类别 1 的概率
precision, recall, _ = precision_recall_curve(labels[val_indices].numpy(), probs)
pr_auc = auc(recall, precision)

plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Category 1 (Purchase Intention)')
plt.legend()
plt.show()