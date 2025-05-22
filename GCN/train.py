import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Carrega o grafo GML
G_nx = nx.read_gml("out/grapphh.gml")

node_mapping = {node: idx for idx, node in enumerate(G_nx.nodes)}

# Extrai features e rótulos dos nós válidos
feature_keys = [
    "num_transacoes_feitas",
    "num_transacoes_recebidas",
    "valor_medio_enviado",
    "valor_medio_recebido",
    "desvio_padrao_transacoes",
    "num_contas_relacionadas",
    "num_estornos",
    "num_cancelamentos"
]

x = []
raw_labels = []
valid_nodes = []

for node in G_nx.nodes(data=True):
    node_attr = node[1]
    if node_attr.get("level") is not None:
        features = [float(node_attr.get(k, 0.0)) for k in feature_keys]
        x.append(features)
        raw_labels.append(node_attr["level"])
        valid_nodes.append(node_mapping[node[0]])

x = torch.tensor(x, dtype=torch.float)

label_encoder = LabelEncoder()
y = torch.tensor(label_encoder.fit_transform(raw_labels), dtype=torch.long)

# Dividir os nós válidos em treino e teste
train_idx, test_idx = train_test_split(range(len(valid_nodes)), test_size=0.2, random_state=42)
train_nodes = set(valid_nodes[i] for i in train_idx)
test_nodes = set(valid_nodes[i] for i in test_idx)

# Criar masks
train_mask = torch.zeros(len(valid_nodes), dtype=torch.bool)
test_mask = torch.zeros(len(valid_nodes), dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True

# Filtrar arestas que ligam apenas treino<->treino ou teste<->teste
edge_index_all = []
for src, dst in G_nx.edges:
    src_idx = node_mapping[src]
    dst_idx = node_mapping[dst]
    if src_idx in valid_nodes and dst_idx in valid_nodes:
        if (src_idx in train_nodes and dst_idx in train_nodes) or \
           (src_idx in test_nodes and dst_idx in test_nodes):
            edge_index_all.append([valid_nodes.index(src_idx), valid_nodes.index(dst_idx)])

edge_index = torch.tensor(edge_index_all, dtype=torch.long).t().contiguous()

# Criar o objeto PyG final
data = Data(x=x, edge_index=edge_index, y=y)
data.train_mask = train_mask
data.test_mask = test_mask

# Modelo GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(data.num_features, 32, len(torch.unique(y))).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

model.eval()
pred = out.argmax(dim=1)
correct = pred[data.test_mask] == data.y[data.test_mask]
acc = int(correct.sum()) / int(data.test_mask.sum())
print(f'Test Accuracy: {acc:.4f}')
