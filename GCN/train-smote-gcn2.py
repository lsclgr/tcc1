import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, BatchNorm1d
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import numpy as np
from collections import Counter

# Carregamento do grafo
G_nx = nx.read_gml("out/grapphh.gml")
node_mapping = {node: idx for idx, node in enumerate(G_nx.nodes)}
edges = [[node_mapping[u], node_mapping[v]] for u, v in G_nx.edges]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Features e labels
feature_keys = [
    "num_transacoes_feitas", "num_transacoes_recebidas", "valor_medio_enviado",
    "valor_medio_recebido", "desvio_padrao_transacoes", "num_contas_relacionadas",
    "num_estornos", "num_cancelamentos"
]

x, y_raw, valid_nodes = [], [], []
for node, attr in G_nx.nodes(data=True):
    if attr.get("level") is not None:
        x.append([float(attr.get(k, 0.0)) for k in feature_keys])
        y_raw.append(attr["level"])
        valid_nodes.append(node_mapping[node])

x = torch.tensor(x, dtype=torch.float)
y = torch.tensor(LabelEncoder().fit_transform(y_raw), dtype=torch.long)

# edge_index limpo
valid_set = set(valid_nodes)
mask = [(src, dst) for src, dst in edge_index.t().tolist() if src in valid_set and dst in valid_set]
edge_index = torch.tensor(mask, dtype=torch.long).t().contiguous()

# Máscaras treino/teste
train_idx, test_idx = train_test_split(range(x.shape[0]), test_size=0.2, random_state=42)
train_mask = torch.zeros(x.shape[0], dtype=torch.bool)
test_mask = torch.zeros(x.shape[0], dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True

# Dados para PyG
data = Data(x=x, edge_index=edge_index, y=y)
data.train_mask = train_mask
data.test_mask = test_mask

# Modelo Profundo
class DeepRobustGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.norms.append(BatchNorm1d(hidden_channels))

        for _ in range(5):  # total 6 camadas GCN
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(BatchNorm1d(hidden_channels))

        self.dropout = Dropout(p=0.6)
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        return self.classifier(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepRobustGCN(x.size(1), 128, len(torch.unique(y))).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

# Calcula pesos das classes para balancear a perda
labels_train = y[train_mask]
class_counts = Counter(labels_train.tolist())
num_classes = y.max().item() + 1
total = sum(class_counts.values())

class_weights = []
for i in range(num_classes):
    count = class_counts.get(i, 0)
    if count > 0:
        class_weights.append(total / (num_classes * count))
    else:
        class_weights.append(0.0)

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

best_loss = float("inf")
patience = 30
counter = 0

for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if loss.item() < best_loss:
        best_loss = loss.item()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}")

# Avaliação
model.eval()
pred = model(data).argmax(dim=1)
correct = pred[data.test_mask] == data.y[data.test_mask]
acc = int(correct.sum()) / int(data.test_mask.sum())
print(f"Test Accuracy: {acc:.4f}")
