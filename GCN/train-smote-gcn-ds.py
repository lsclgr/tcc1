import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, BatchNorm1d, ModuleList
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import networkx as nx
import numpy as np
from collections import Counter
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Pré-processamento de Dados ================================================

def load_and_preprocess_data(file_path):
    """Carrega e pré-processa os dados do grafo"""
    # Carregamento do grafo
    G_nx = nx.read_gml(file_path)
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

    # Normalização das features
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(LabelEncoder().fit_transform(y_raw), dtype=torch.long)

    # Filtro de nós válidos
    valid_set = set(valid_nodes)
    mask = [(src, dst) for src, dst in edge_index.t().tolist() if src in valid_set and dst in valid_set]
    edge_index = torch.tensor(mask, dtype=torch.long).t().contiguous()

    return x, y, edge_index, valid_nodes

# 2. Definição do Modelo ======================================================

class EdgeDropout(torch.nn.Module):
    """Regularização por dropout nas arestas do grafo"""
    def __init__(self, p=0.3):
        super().__init__()
        self.p = p
        
    def forward(self, edge_index):
        if self.training and self.p > 0:
            mask = torch.rand(edge_index.size(1)) >= self.p
            return edge_index[:, mask]
        return edge_index

class DeepRobustGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = ModuleList()
        self.norms = ModuleList()
        self.drops = ModuleList()
        self.edge_dropout = EdgeDropout(p=0.3)
        
        # Camadas com dimensões decrescentes
        hidden_dims = [hidden_channels] * 3 + [hidden_channels//2] * 2 + [hidden_channels//4]
        
        # Primeira camada
        self.convs.append(GCNConv(in_channels, hidden_dims[0]))
        self.norms.append(BatchNorm1d(hidden_dims[0]))
        self.drops.append(Dropout(p=0.5))
        
        # Camadas intermediárias
        for i in range(len(hidden_dims)-1):
            self.convs.append(GCNConv(hidden_dims[i], hidden_dims[i+1]))
            self.norms.append(BatchNorm1d(hidden_dims[i+1]))
            self.drops.append(Dropout(p=0.5 if i < 3 else 0.3))
        
        # Camada final
        self.classifier = Linear(hidden_dims[-1], out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        if hasattr(self.classifier, 'reset_parameters'):
            self.classifier.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Aplica dropout nas arestas durante o treino
        if self.training:
            edge_index = self.edge_dropout(edge_index)
        
        for conv, norm, drop in zip(self.convs, self.norms, self.drops):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.leaky_relu(x, negative_slope=0.2)
            x = drop(x)
        return self.classifier(x)

# 3. Treinamento e Avaliação ===================================================

def train_model(data, model, optimizer, scheduler, class_weights, patience=30):
    best_val_loss = float('inf')
    counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
        
        # Regularização L2 adicional
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss += 1e-4 * l2_reg
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        
        # Validação
        model.eval()
        with torch.no_grad():
            val_out = model(data)
            val_loss = F.cross_entropy(val_out[data.val_mask], data.y[data.val_mask])
        
        scheduler.step(val_loss)
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, LR: {current_lr:.6f}")
    
    # Carrega o melhor modelo
    model.load_state_dict(torch.load('best_model.pt'))
    return model, history

def evaluate_model(model, data, save_dir="resultGCN"):
    # Cria o diretório se não existir
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        y_true = data.y[data.test_mask].cpu()
        y_pred = pred[data.test_mask].cpu()
        
        # Acurácia
        correct = y_pred == y_true
        acc = int(correct.sum()) / int(data.test_mask.sum())
        
        # Relatório de classificação
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        # Matriz de confusão numérica
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix (Números):")
        print(cm)
        
        # Plot da matriz de confusão
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=np.unique(y_true),
                        yticklabels=np.unique(y_true))
        ax.set_title('Matriz de Confusão')
        ax.set_ylabel('Verdadeiro')
        ax.set_xlabel('Predito')
        
        # Salva a figura
        cm_path = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"\nMatriz de Confusão salva em: {cm_path}")
        
        # Salva também os valores numéricos em um arquivo .txt
        txt_path = os.path.join(save_dir, 'confusion_matrix.txt')
        np.savetxt(txt_path, cm, fmt='%d')
        print(f"Valores numéricos salvos em: {txt_path}")
        
        return acc

# 4. Fluxo Principal ==========================================================

if __name__ == "__main__":
    # Configuração inicial
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Carrega e prepara os dados
    x, y, edge_index, valid_nodes = load_and_preprocess_data("out/grafo_fraude_c.gml")
    
    # Cria o objeto Data do PyG
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # Divisão estratificada dos dados
    train_idx, test_idx = train_test_split(
        range(len(valid_nodes)), 
        test_size=0.2, 
        random_state=42, 
        stratify=y.numpy()
    )
    train_idx, val_idx = train_test_split(
        train_idx, 
        test_size=0.2, 
        random_state=42, 
        stratify=y[train_idx].numpy()
    )
    
    # Cria as máscaras
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True
    
    # Move os dados para o dispositivo
    data = data.to(device)
    
    # Balanceamento de classes
    labels_train = y[train_idx]
    class_counts = Counter(labels_train.tolist())
    num_classes = y.max().item() + 1
    total = sum(class_counts.values())
    
    class_weights = torch.tensor([
        total / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)
    ], dtype=torch.float32).to(device)
    
    # Inicializa o modelo
    model = DeepRobustGCN(
        in_channels=data.num_features,
        hidden_channels=128,
        out_channels=num_classes
    ).to(device)
    
    # Configura otimizador e scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.01, 
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=10
    )
    
    # Treinamento
    print("Starting training...")
    model, history = train_model(data, model, optimizer, scheduler, class_weights)
    
    # Avaliação
    test_acc = evaluate_model(model, data)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")