import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, BatchNorm
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# === Carrega o grafo ===
G_nx = nx.read_gml("out/grapphh.gml")
node_mapping = {node: idx for idx, node in enumerate(G_nx.nodes)}
edge_index = torch.tensor(
    [[node_mapping[src], node_mapping[dst]] for src, dst in G_nx.edges],
    dtype=torch.long
).t().contiguous()

# === Define as features e rótulos ===
feature_keys = [
    "num_transacoes_feitas", "num_transacoes_recebidas",
    "valor_medio_enviado", "valor_medio_recebido",
    "desvio_padrao_transacoes", "num_contas_relacionadas",
    "num_estornos", "num_cancelamentos"
]

x, y_raw, valid_nodes = [], [], []
for node, attrs in G_nx.nodes(data=True):
    if attrs.get("level") is not None:
        x.append([float(attrs.get(k, 0.0)) for k in feature_keys])
        y_raw.append(attrs["level"])
        valid_nodes.append(node_mapping[node])

x = torch.tensor(x, dtype=torch.float)
label_encoder = LabelEncoder()
y = torch.tensor(label_encoder.fit_transform(y_raw), dtype=torch.long)

# === Filtra arestas com base nos nós válidos ===
valid_set = set(valid_nodes)
edge_index = torch.tensor([
    [src, dst] for src, dst in edge_index.t().tolist()
    if src in valid_set and dst in valid_set
], dtype=torch.long).t().contiguous()

# === Divide treino/teste ===
train_idx, test_idx = train_test_split(
    np.arange(x.shape[0]), test_size=0.2, stratify=y, random_state=42
)
train_mask = torch.zeros(x.size(0), dtype=torch.bool)
test_mask = torch.zeros(x.size(0), dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True

# === Aplica SMOTE apenas nos dados de treino ===
x_train = x[train_mask].numpy()
y_train = y[train_mask].numpy()
smote = SMOTE(random_state=42, k_neighbors=2)
x_resampled, y_resampled = smote.fit_resample(x_train, y_train)

# === Junta com dados de teste ===
x_total = torch.tensor(np.vstack([x_resampled, x[test_mask].numpy()]), dtype=torch.float)
y_total = torch.tensor(np.concatenate([y_resampled, y[test_mask].numpy()]), dtype=torch.long)
resampled_train_mask = torch.zeros(x_total.size(0), dtype=torch.bool)
resampled_train_mask[:len(x_resampled)] = True
resampled_test_mask = torch.zeros(x_total.size(0), dtype=torch.bool)
resampled_test_mask[len(x_resampled):] = True

# === Dados PyG ===
data = Data(x=x_total, edge_index=edge_index, y=y_total)
data.train_mask = resampled_train_mask
data.test_mask = resampled_test_mask

# === Modelo Robusto ===
class RobustGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden1, hidden2, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden1)
        self.bn1 = BatchNorm(hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.bn2 = BatchNorm(hidden2)
        self.conv3 = GCNConv(hidden2, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return x

# === Treinamento ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RobustGCN(data.num_features, 64, 32, len(torch.unique(y_total))).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# === Avaliação ===
model.eval()
pred = model(data).argmax(dim=1).cpu().numpy()
true = data.y.cpu().numpy()
mask = data.test_mask.cpu().numpy()

print(f"\nTest Accuracy: {(pred[mask] == true[mask]).mean():.4f}")
print("\nClassification Report:")
print(classification_report_imbalanced(true[mask], pred[mask], target_names=label_encoder.classes_))
