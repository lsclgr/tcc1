import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, BatchNorm1d
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import networkx as nx
import numpy as np

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
y_raw = np.array(y_raw)
label_encoder = LabelEncoder()
y = torch.tensor(label_encoder.fit_transform(y_raw), dtype=torch.long)

# Filtra arestas para manter apenas nós válidos
valid_set = set(valid_nodes)
mask = [(src, dst) for src, dst in edge_index.t().tolist() if src in valid_set and dst in valid_set]
edge_index = torch.tensor(mask, dtype=torch.long).t().contiguous()

# Divisão em treino, validação e teste
train_idx, test_idx = train_test_split(range(x.shape[0]), test_size=0.2, random_state=42, stratify=y_raw)
train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=42, stratify=y_raw[train_idx])

train_mask = torch.zeros(x.shape[0], dtype=torch.bool)
val_mask = torch.zeros(x.shape[0], dtype=torch.bool)
test_mask = torch.zeros(x.shape[0], dtype=torch.bool)
train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

# Dados para PyG
data = Data(x=x, edge_index=edge_index, y=y)
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

# Calcula pesos das classes para o loss
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y.numpy()),
    y=y.numpy()
)
class_weights = torch.tensor(class_weights, dtype=torch.float)

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

# Treinamento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepRobustGCN(x.size(1), 128, len(torch.unique(y))).to(device)
data = data.to(device)
class_weights = class_weights.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

best_val_loss = float("inf")
patience = 30
counter = 0

for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss_train = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
    loss_train.backward()
    optimizer.step()
    scheduler.step()

    model.eval()
    with torch.no_grad():
        out = model(data)
        loss_val = F.cross_entropy(out[data.val_mask], data.y[data.val_mask], weight=class_weights)

    if loss_val.item() < best_val_loss:
        best_val_loss = loss_val.item()
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Train Loss: {loss_train.item():.4f}, Val Loss: {loss_val.item():.4f}")

# Avaliação final com melhor modelo
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
with torch.no_grad():
    pred = model(data).argmax(dim=1)
    correct = pred[data.test_mask] == data.y[data.test_mask]
    acc = int(correct.sum()) / int(data.test_mask.sum())
print(f"Test Accuracy: {acc:.4f}")
