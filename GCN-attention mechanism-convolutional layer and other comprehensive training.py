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
import matplotlib.pyplot as plt
import seaborn as sns
import random

# 设置随机种子以确保结果的可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 保证每次运行的确定性
    torch.backends.cudnn.benchmark = False

# 调用设置随机种子函数
set_seed(42)

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
        self.affective_attention = AffectiveAttentionLayer(affective_dim)
        self.cognitive_attention = CognitiveAttentionLayer(cognitive_dim)
        self.conv1 = GCNConv(affective_dim + cognitive_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        affective_x = x[:, affective_indices]
        cognitive_x = x[:, cognitive_indices]

        affective_x = self.affective_attention(affective_x)
        cognitive_x = self.cognitive_attention(cognitive_x)

        x = torch.cat([affective_x, cognitive_x], dim=1)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 实例化模型
num_affective_features = len(affective_indices)
num_cognitive_features = len(cognitive_indices)
num_classes = 2
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

# 注意力权重的可视化
def plot_attention_weights(attention_weights, title, labels):
    plt.figure(figsize=(8, 4))
    weights = attention_weights.detach().numpy().squeeze()
    plt.bar(labels, weights, color='skyblue')
    plt.title(title)
    plt.ylabel('Attention Score')
    plt.xlabel('Feature')
    plt.show()

# 情绪变量注意力权重
affective_attention_weights = model.affective_attention.attention_weights
plot_attention_weights(affective_attention_weights, 'Affective Attention Weights', ['PL', 'IPS', 'ANP'])

# 认知变量注意力权重
cognitive_attention_weights = model.cognitive_attention.attention_weights
plot_attention_weights(cognitive_attention_weights, 'Cognitive Attention Weights', ['IMM', 'ARIT'])

# 热图展示注意力权重
def plot_attention_heatmap(attention_weights, title, labels):
    plt.figure(figsize=(6, 2))
    weights = attention_weights.detach().numpy().squeeze()
    sns.heatmap(weights.reshape(1, -1), annot=True, cmap="YlGnBu", cbar=False, xticklabels=labels)
    plt.title(title)
    plt.show()

# 热图形式展示情绪变量的注意力权重
plot_attention_heatmap(affective_attention_weights, 'Affective Attention Heatmap', ['PL', 'IPS', 'ANP'])

# 热图形式展示认知变量的注意力权重
plot_attention_heatmap(cognitive_attention_weights, 'Cognitive Attention Heatmap', ['IMM', 'ARIT'])

# 层间权重的可视化
def plot_layer_weights(layer_weights, title):
    weights = layer_weights.detach().numpy()
    plt.figure(figsize=(8, 6))
    sns.heatmap(weights, annot=True, cmap="YlGnBu", cbar=True)
    plt.title(title)
    plt.show()

# 可视化 GCN 层的权重
plot_layer_weights(model.conv1.lin.weight, 'GCN Layer 1 Weights')
plot_layer_weights(model.conv2.lin.weight, 'GCN Layer 2 Weights')

# 可视化神经元激活模式
def visualize_activation(x, title):
    plt.figure(figsize=(8, 4))
    activation = x.detach().numpy()
    plt.plot(activation)
    plt.title(title)
    plt.ylabel('Activation')
    plt.xlabel('Neuron Index')
    plt.show()

# 获取中间层激活模式
model.eval()
with torch.no_grad():
    affective_x = model.affective_attention(graph_data.x[:, affective_indices])
    cognitive_x = model.cognitive_attention(graph_data.x[:, cognitive_indices])
    merged_x = torch.cat([affective_x, cognitive_x], dim=1)
    hidden_x = model.conv1(merged_x, graph_data.edge_index)

    # 可视化情绪注意力输出
    visualize_activation(affective_x, "Affective Attention Output")

    # 可视化认知注意力输出
    visualize_activation(cognitive_x, "Cognitive Attention Output")

    # 可视化第一个 GCN 卷积层的输出
    visualize_activation(hidden_x, "GCN Layer 1 Activation Output")

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(200), train_loss_values, label='Train Loss', color='blue')
plt.plot(range(200), val_loss_values, label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss with Attention Mechanism')
plt.legend()
plt.show()









































































