import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchbnn as bnn  # 贝叶斯神经网络库
import numpy as np
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

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

# 选择潜变量得分列（PL、IPS、ANP、IMM、ARIT）
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

# 定义贝叶斯图卷积网络
class BayesianGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(BayesianGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        # 贝叶斯神经网络层，输出不确定性
        self.bayesian_layer = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim, out_features=num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.bayesian_layer(x)  # 贝叶斯线性层
        return F.log_softmax(x, dim=1)

# 实例化贝叶斯模型
model = BayesianGCN(num_features=5, hidden_dim=16, num_classes=2)  # 5个自变量

# 使用交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss()
labels = torch.tensor(data['PI_category'].values)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.8, 0.999))

# 贝叶斯网络通过多次采样输出预测概率
def bayesian_prediction(model, data, num_samples=10):
    model.eval()
    predictions = []
    for _ in range(num_samples):
        with torch.no_grad():
            out = model(data)
            predictions.append(F.softmax(out, dim=1).cpu().numpy())
    predictions = np.array(predictions)
    # 平均所有样本的预测
    mean_predictions = np.mean(predictions, axis=0)
    return mean_predictions

# 训练模型
train_loss_values = []
val_loss_values = []
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(graph_data)
    loss = criterion(out[train_indices], labels[train_indices])
    loss.backward()
    optimizer.step()

    # 记录损失
    train_loss_values.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_out = model(graph_data)
        val_loss = criterion(val_out[val_indices], labels[val_indices])
        val_loss_values.append(val_loss.item())

    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}')

# 获取概率化预测
probs = bayesian_prediction(model, graph_data)

# 使用 t-SNE 或 UMAP 进行降维
method = 'tsne'  # 可以选择 'tsne' 或 'umap'
deep_features_np = out.cpu().detach().numpy()

if method == 'tsne':
    print("Using t-SNE for dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(deep_features_np)
elif method == 'umap':
    print("Using UMAP for dimensionality reduction...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced_features = reducer.fit_transform(deep_features_np)

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels.numpy(), cmap='coolwarm', label='Low Purchase Intention')
plt.colorbar()
plt.title(f'Dimensionality Reduction using {method.upper()}')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()

# 输出每个节点的预测类别和不确定性
predicted_classes = np.argmax(probs, axis=1)
uncertainty = np.std(probs, axis=0)

for i, (pred, uncert) in enumerate(zip(predicted_classes, uncertainty)):
    print(f'Node {i}: Predicted Class = {pred}, Uncertainty = {uncert}')