import torch
import numpy as np
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import networkx as nx
from torch_geometric.utils import from_networkx
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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
        # 返回第一层卷积的特征（深度特征）
        return x  # 提取中间层特征

# 实例化模型
model = GCN(num_features=5, hidden_dim=16, num_classes=2)

# 使用交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss()
labels = torch.tensor(data['PI_category'].values)

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.8, 0.999))

# 训练模型（简化训练过程）
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(graph_data)
    loss = criterion(out[train_indices], labels[train_indices])
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_out = model(graph_data)
        val_loss = criterion(val_out[val_indices], labels[val_indices])

# 深度特征提取
model.eval()
with torch.no_grad():
    deep_features = model(graph_data)  # 提取中间层特征

# 将特征转换为 NumPy 数组
deep_features_np = deep_features.cpu().numpy()

# 将分类标签也转换为 NumPy 数组
labels_np = labels.cpu().numpy()

# 选择 t-SNE 或 UMAP 进行降维
method = 'tsne'  # 可以选择 'tsne' 或 'umap'

if method == 'tsne':
    print("Using t-SNE for dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(deep_features_np)
elif method == 'umap':
    print("Using UMAP for dimensionality reduction...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced_features = reducer.fit_transform(deep_features_np)

# 可视化降维后的结果
plt.figure(figsize=(10, 6))

# 根据标签可视化不同的类
plt.scatter(reduced_features[labels_np == 0, 0], reduced_features[labels_np == 0, 1],
            c='blue', label='Low Purchase Intention', alpha=0.6)
plt.scatter(reduced_features[labels_np == 1, 0], reduced_features[labels_np == 1, 1],
            c='red', label='High Purchase Intention', alpha=0.6)

plt.title(f'Dimensionality Reduction using {method.upper()}')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()
plt.show()