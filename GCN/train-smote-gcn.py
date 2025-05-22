import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Carrega o grafo GML
G_nx = nx.read_gml("out/grapphh.gml")

# Mapeia os nós para índices
node_mapping = {node: idx for idx, node in enumerate(G_nx.nodes)}

# edge_index
edge_index = []
for src, dst in G_nx.edges:
    edge_index.append([node_mapping[src], node_mapping[dst]])
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Features e labels
feature_keys = [
    "num_transacoes_feitas", "num_transacoes_recebidas",
    "valor_medio_enviado", "valor_medio_recebido",
    "desvio_padrao_transacoes", "num_contas_relacionadas",
    "num_estornos", "num_cancelamentos"
]

x, raw_labels, valid_nodes = [], [], []

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

# Ajusta edge_index com nós válidos
valid_node_set = set(valid_nodes)
mask = [(src, dst) for src, dst in edge_index.t().tolist() if src in valid_node_set and dst in valid_node_set]
edge_index = torch.tensor(mask, dtype=torch.long).t().contiguous()

# Split treino/teste
train_idx, test_idx = train_test_split(range(x.shape[0]), test_size=0.2, random_state=42)
x_train, y_train = x[train_idx], y[train_idx]
x_test, y_test = x[test_idx], y[test_idx]

# SMOTE no treino
smote = SMOTE(k_neighbors=5)
x_resampled, y_resampled = smote.fit_resample(x_train.cpu().numpy(), y_train.cpu().numpy())
x_resampled = torch.tensor(x_resampled, dtype=torch.float)
y_resampled = torch.tensor(y_resampled, dtype=torch.long)

# Junta com os dados de teste
x_final = torch.cat([x_resampled, x_test], dim=0)
y_final = torch.cat([y_resampled, y_test], dim=0)

# Máscaras
num_train = len(x_resampled)
num_total = x_final.size(0)
train_mask = torch.zeros(num_total, dtype=torch.bool)
test_mask = torch.zeros(num_total, dtype=torch.bool)
train_mask[:num_train] = True
test_mask[num_train:] = True

# edge_index mantido igual (opcional: conectar sintéticos com k-NN ou rede fictícia)
data = Data(x=x_final, y=y_final, edge_index=edge_index)
data.train_mask = train_mask
data.test_mask = test_mask

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
