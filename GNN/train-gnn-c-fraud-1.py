import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np

# --- Assumindo que 'data' (o objeto torch_geometric.data.Data)
# --- e suas máscaras (data.train_mask, data.val_mask, data.test_mask)
# --- foram criados e estão disponíveis a partir das etapas anteriores.
# --- Como o código não pôde ser executado, 'data' não está disponível.
# --- A parte de "feature engineering" e "labeling" já foi concluída com sucesso
# --- e `node_features_scaled`, `node_labels_np`, `edge_index_np`, `edge_attributes_scaled` estão disponíveis.

# Re-criando o objeto 'data' para demonstração do modelo e treinamento,
# já que não pudemos criá-lo na execução anterior.
# Isso é para fins de demonstração do código completo que você deve executar.

# Supondo que você tenha node_features_scaled, node_labels_np, edge_index_np, edge_attributes_scaled
# do passo anterior.
# Estes seriam convertidos em tensores PyTorch
x_demo = torch.tensor(node_features_scaled, dtype=torch.float)
y_demo = torch.tensor(node_labels_np, dtype=torch.long)
edge_index_demo = torch.tensor(edge_index_np, dtype=torch.long)
edge_attr_demo = torch.tensor(edge_attributes_scaled, dtype=torch.float)

# Crie um objeto Data de exemplo para que o código do modelo possa ser definido
# No seu ambiente, você usaria o objeto 'data' real gerado anteriormente.
data_demo = Data(x=x_demo, edge_index=edge_index_demo, edge_attr=edge_attr_demo, y=y_demo)

# Criar máscaras de exemplo para simular o que seria feito
num_nodes_demo = data_demo.num_nodes
all_indices_demo = np.arange(num_nodes_demo)
train_idx_demo, temp_idx_demo = train_test_split(all_indices_demo, test_size=0.4, stratify=data_demo.y.numpy(), random_state=42)
val_idx_demo, test_idx_demo = train_test_split(temp_idx_demo, test_size=0.5, stratify=data_demo.y[temp_idx_demo].numpy(), random_state=42)

train_mask_demo = torch.zeros(num_nodes_demo, dtype=torch.bool)
val_mask_demo = torch.zeros(num_nodes_demo, dtype=torch.bool)
test_mask_demo = torch.zeros(num_nodes_demo, dtype=torch.bool)

train_mask_demo[train_idx_demo] = True
val_mask_demo[val_idx_demo] = True
test_mask_demo[test_idx_demo] = True

data_demo.train_mask = train_mask_demo
data_demo.val_mask = val_mask_demo
data_demo.test_mask = test_mask_demo


# 2. Definição do Modelo GNN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # GCNConv é uma camada de convolução de grafo que agrega informações dos vizinhos
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Primeira camada GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x) # Função de ativação ReLU
        x = F.dropout(x, p=0.5, training=self.training) # Dropout para regularização

        # Segunda camada GCN
        x = self.conv2(x, edge_index)
        return x

# Configurações do modelo
# data.num_node_features seria `x_demo.shape[1]`
# data.y.unique().size(0) seria `y_demo.unique().size(0)`
in_channels = data_demo.num_node_features # Número de características de entrada por nó
hidden_channels = 64 # Número de características na camada oculta
out_channels = data_demo.y.unique().size(0) # Número de classes de saída (2 para fraude/legítimo)

model = GCN(in_channels, hidden_channels, out_channels)
print(f"\nModelo GNN definido:\n{model}")

# 3. Configuração do Treinamento
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) # Otimizador Adam
criterion = torch.nn.CrossEntropyLoss() # Função de perda para classificação multiclasse

def train():
    model.train() # Define o modelo para o modo de treinamento
    optimizer.zero_grad() # Zera os gradientes
    out = model(data_demo.x, data_demo.edge_index) # Forward pass
    loss = criterion(out[data_demo.train_mask], data_demo.y[data_demo.train_mask]) # Calcula a perda apenas para os nós de treinamento
    loss.backward() # Backward pass
    optimizer.step() # Atualiza os pesos do modelo
    return loss.item()

def evaluate(mask):
    model.eval() # Define o modelo para o modo de avaliação
    with torch.no_grad(): # Desabilita o cálculo de gradientes
        out = model(data_demo.x, data_demo.edge_index)
        pred = out[mask].argmax(dim=1) # Obtém a classe prevista
        labels = data_demo.y[mask]

        accuracy = accuracy_score(labels.cpu().numpy(), pred.cpu().numpy())
        f1 = f1_score(labels.cpu().numpy(), pred.cpu().numpy(), average='binary') # 'binary' para classificação binária
        precision = precision_score(labels.cpu().numpy(), pred.cpu().numpy(), average='binary')
        recall = recall_score(labels.cpu().numpy(), pred.cpu().numpy(), average='binary')
    return accuracy, f1, precision, recall

# 4. Ciclo de Treinamento
epochs = 200 # Número de épocas de treinamento

print("\nIniciando treinamento do modelo GNN...")
for epoch in range(1, epochs + 1):
    loss = train()
    if epoch % 10 == 0 or epoch == epochs:
        train_acc, train_f1, train_prec, train_rec = evaluate(data_demo.train_mask)
        val_acc, val_f1, val_prec, val_rec = evaluate(data_demo.val_mask)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
              f'Val F1: {val_f1:.4f}, Val Prec: {val_prec:.4f}, Val Rec: {val_rec:.4f}')

# 5. Avaliação Final no conjunto de Teste
print("\nAvaliação final no conjunto de Teste:")
test_acc, test_f1, test_prec, test_rec = evaluate(data_demo.test_mask)
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Test F1 Score: {test_f1:.4f}')
print(f'Test Precision: {test_prec:.4f}')
print(f'Test Recall: {test_rec:.4f}')