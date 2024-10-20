import torch
import numpy as np
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import networkx as nx
from torch_geometric.utils import from_networkx
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
import torch
import numpy as np
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import networkx as nx
from torch_geometric.utils import from_networkx
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv

# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)  # 设置随机种子

# 读取潜变量得分数据
data = pd.read_csv('/Users/lirui/Desktop/4.csv')
print(data.head())

# 选择潜变量得分列
latent_scores = data[['PL', 'IPS', 'ANP', 'IMM', 'ARIT']].values

# 计算余弦相似度
similarity_matrix = cosine_similarity(latent_scores)

# 生成邻接矩阵并创建图
threshold = 0.85
adjacency_matrix = (similarity_matrix > threshold).astype(int)
G = nx.from_numpy_array(adjacency_matrix)

# 为每个节点添加潜变量得分作为特征
for i, node in enumerate(G.nodes):
    G.nodes[node]['feature'] = latent_scores[i]

graph_data = from_networkx(G)
graph_data.x = torch.tensor(latent_scores, dtype=torch.float)
print(graph_data)

# 转换 PI 分数为分类标签
data['PI_category'] = data['PI'].apply(lambda pi_value: 0 if pi_value <= 17 else 1)
print(data['PI_category'].unique())

train_indices, val_indices = train_test_split(range(graph_data.num_nodes), test_size=0.2, random_state=42)
train_indices = torch.tensor(train_indices, dtype=torch.long)
val_indices = torch.tensor(val_indices, dtype=torch.long)


# 模型定义
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# 实例化模型
model = GCN(num_features=5, hidden_dim=16, num_classes=2)

# 使用交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss()
labels = torch.tensor(data['PI_category'].values)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.8, 0.999))

# 使用学习率调度器，每 50 个 epoch 将学习率减少 10 倍
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# 训练模型
train_loss_values = []
val_loss_values = []
train_acc_values = []  # 保存训练集的准确率
val_acc_values = []  # 保存验证集的准确率

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(graph_data)
    loss = criterion(out[train_indices], labels[train_indices])
    loss.backward()
    optimizer.step()

    # 计算训练集的准确率
    pred_train = out[train_indices].max(1)[1]  # 预测最大概率的类别
    correct_train = pred_train.eq(labels[train_indices]).sum().item()  # 计算正确预测的数量
    train_acc = correct_train / len(train_indices)  # 计算训练集准确率
    train_loss_values.append(loss.item())
    train_acc_values.append(train_acc)  # 保存训练集的准确率

    model.eval()
    with torch.no_grad():
        val_out = model(graph_data)
        val_loss = criterion(val_out[val_indices], labels[val_indices])
        val_loss_values.append(val_loss.item())

        # 计算验证集的准确率
        pred_val = val_out[val_indices].max(1)[1]  # 预测最大概率的类别
        correct_val = pred_val.eq(labels[val_indices]).sum().item()  # 计算正确预测的数量
        val_acc = correct_val / len(val_indices)  # 计算验证集准确率
        val_acc_values.append(val_acc)  # 保存验证集的准确率

    if epoch % 20 == 0:
        print(
            f'Epoch {epoch}, Train Loss: {loss.item()}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss.item()}, Val Acc: {val_acc:.4f}')

# 绘制训练和验证损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(200), train_loss_values, label='Train Loss', color='blue')
plt.plot(range(200), val_loss_values, label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.show()

# 生成邻接矩阵并创建图
threshold = 0.95
adjacency_matrix = (similarity_matrix > threshold).astype(int)
G = nx.from_numpy_array(adjacency_matrix)

# 计算图结构指标
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
average_degree = 2 * num_edges / num_nodes
graph_density = nx.density(G)

# 为每个节点添加潜变量得分作为特征
for i, node in enumerate(G.nodes):
    G.nodes[node]['feature'] = latent_scores[i]

G.remove_edges_from(nx.selfloop_edges(G))
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
average_degree = (2 * num_edges) / num_nodes
graph_density = (2 * num_edges) / (num_nodes * (num_nodes - 1))

print(f"图结构概要：")
print(f"- 节点数量：{num_nodes}")
print(f"- 边的数量：{num_edges}")
print(f"- 平均度：{average_degree:.2f}")
print(f"- 图密度：{graph_density:.4f}")