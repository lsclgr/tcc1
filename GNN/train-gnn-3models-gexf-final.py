import argparse
import os
import sys
import re
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
import matplotlib.pyplot as plt

# Importar das bibliotecas de GNN
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv, GCNConv
from torch.nn import Linear, Dropout
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- 1. Definição dos Modelos ---

class GCN(torch.nn.Module):
    """Modelo Graph Convolutional Network (GCN)."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = Dropout(p=dropout)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphSAGE(torch.nn.Module):
    """Modelo GraphSAGE."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = Dropout(p=dropout)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    """Modelo Graph Attention Network (GAT)."""
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels * heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# --- 2. Gerenciador de Logs ---

class Logger(object):
    """Redireciona a saída do print para o console e um arquivo."""
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# --- 3. Funções de Pré-processamento e Balanceamento ---

def load_and_preprocess_data(gexf_path):
    """
    Carrega e pré-processa os dados de um arquivo GEXF.
    Esta função foi modificada para ler o formato GEXF e extrair features
    diretamente dos atributos dos nós, que já estão estruturados.
    """
    print("Carregando o grafo do arquivo GEXF...")
    try:
        # MODIFICAÇÃO: Alterado de read_gml para read_gexf
        G = nx.read_gexf(gexf_path)
    except Exception as e:
        print(f"Erro ao ler o arquivo GEXF: {e}")
        return None

    print("Pré-processando os dados dos nós...")
    node_ids = sorted(list(G.nodes()))
    # Extrai os atributos dos nós para um DataFrame
    df_nodes = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    
    # MODIFICAÇÃO: A coluna 'level' continua sendo o alvo
    # Garante que a coluna 'level' exista antes de prosseguir
    if 'level' not in df_nodes.columns:
        print("Erro: A coluna 'level' (alvo) não foi encontrada nos dados dos nós.")
        return None
        
    df_nodes['target'] = df_nodes['level'].apply(lambda x: 1 if x == 'Fraude' else 0)
    
    print("Processando features...")

    # MODIFICAÇÃO: Lógica de extração de features totalmente refeita para o formato GEXF.
    # As funções parse_transactions e parse_account_info não são mais necessárias.
    
    # Identificar colunas numéricas e categóricas
    numeric_features = [
        'profile_created_at', 'profile_birth', 'profile_is_pep', 'profile_is_foreign',
        'profile_length_email', 'profile_name_in_email', 'profile_number_initial_email',
        'profile_number_final_email', 'profile_number_in_email', 'users', 'contact_by_phone',
        'pct_00_05', 'pct_06_11', 'pct_12_17', 'pct_18_23', 'pct_dias_uteis',
        'pct_fins_semana', 'pct_seg', 'pct_ter', 'pct_qua', 'pct_qui', 'pct_sex',
        'pct_sab', 'pct_dom', 'pct_1_10', 'pct_11_20', 'pct_21_31', 'pct_estornos',
        'pct_cancelamentos', 'valor_total_pago', 'valor_medio_pago',
        'valor_total_sem_estorno_cancelamento', 'valor_medio_sem_estorno_cancelamento',
        'qtd_total', 'qtd_total_sem_estorno_cancelamento'
    ]
    
    categorical_features = [
        'documento_tipo', 'profile_career', 'profile_dominio_email'
    ]
    
    # Filtrar features que realmente existem no DataFrame
    available_numeric_features = [f for f in numeric_features if f in df_nodes.columns]
    available_categorical_features = [f for f in categorical_features if f in df_nodes.columns]
    
    print(f"Features numéricas encontradas: {available_numeric_features}")
    print(f"Features categóricas encontradas: {available_categorical_features}")

    # Processar features numéricas
    numeric_df = df_nodes[available_numeric_features].copy()
    for col in numeric_df.columns:
        # Converter para numérico, forçando erros a virarem NaN
        numeric_series = pd.to_numeric(numeric_df[col], errors='coerce')
        # Preencher NaNs com a média da coluna
        mean_val = numeric_series.mean()
        numeric_df[col] = numeric_series.fillna(mean_val)

    # Processar features categóricas
    categorical_df = df_nodes[available_categorical_features].copy()
    # Preencher NaNs com 'missing' para criar uma categoria para eles
    categorical_df.fillna('missing', inplace=True)
    # Criar dummy variables
    dummies_df = pd.get_dummies(categorical_df, prefix=available_categorical_features, dummy_na=False)

    # Concatenar todas as features processadas
    df_final_features = pd.concat([numeric_df.reset_index(drop=True), dummies_df.reset_index(drop=True)], axis=1)
    df_final_features.fillna(0, inplace=True) # Preenchimento final para garantir
    
    print("Escalando features...")
    features_scaled = StandardScaler().fit_transform(df_final_features)
    
    x = torch.tensor(features_scaled, dtype=torch.float)
    y = torch.tensor(df_nodes['target'].values, dtype=torch.long)
    
    # Mapear os IDs de nós (que podem ser strings) para índices inteiros
    node_map = {node_id: i for i, node_id in enumerate(node_ids)}
    edges = [(node_map[u], node_map[v]) for u, v, _ in G.edges(data=True)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=len(node_ids))
    
    print("\n--- Resumo do Pré-processamento ---")
    print(f"Dados processados: {data}\nNúmero de features por nó: {data.num_node_features}\nDistribuição das classes: {torch.bincount(data.y)}")
    print("------------------------------------\n")
    return data

def create_masks(data):
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    
    # Checar se há pelo menos 2 membros em cada classe para estratificação
    if np.sum(data.y.numpy() == 0) < 2 or np.sum(data.y.numpy() == 1) < 2:
        print("Aviso: Não há amostras suficientes para estratificação. Usando divisão simples.")
        train_indices, test_val_indices = train_test_split(indices, test_size=0.3, random_state=42)
        val_indices, test_indices = train_test_split(test_val_indices, test_size=0.5, random_state=42)
    else:
        train_indices, test_val_indices, _, _ = train_test_split(indices, data.y.numpy(), test_size=0.3, random_state=42, stratify=data.y.numpy())
        val_indices, test_indices, _, _ = train_test_split(test_val_indices, data.y.numpy()[test_val_indices], test_size=0.5, random_state=42, stratify=data.y.numpy()[test_val_indices])

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[train_indices] = True
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask[val_indices] = True
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask[test_indices] = True
    
    print(f"Máscaras de dados criadas:\n  - Nós de Treino: {data.train_mask.sum().item()}\n  - Nós de Validação: {data.val_mask.sum().item()}\n  - Nós de Teste: {data.test_mask.sum().item()}")
    return data

# --- 4. Funções de Treinamento e Avaliação ---

def train(model, data, optimizer, criterion, train_indices):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_indices], data.y[train_indices])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data, mask, threshold=0.5):
    model.eval()
    out = model(data.x, data.edge_index)
    probs = out.exp()
    y_true = data.y[mask].cpu().numpy()
    y_prob = probs[mask][:, 1].cpu().numpy()
    y_pred = (y_prob >= threshold).astype(int)
    
    # Evitar erro no roc_auc_score se houver apenas uma classe na máscara
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc
    }
    return metrics, y_true, y_pred

def find_best_threshold(model, data):
    model.eval()
    with torch.no_grad():
        probs = model(data.x, data.edge_index).exp()
    val_probs = probs[data.val_mask][:, 1].cpu().numpy()
    y_val_true = data.y[data.val_mask].cpu().numpy()
    
    best_f1, best_threshold = 0, 0.5
    for threshold in np.arange(0.01, 1.0, 0.01):
        y_pred = (val_probs >= threshold).astype(int)
        f1 = f1_score(y_val_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
            
    print(f"\nMelhor limiar encontrado no conjunto de validação: {best_threshold:.2f} (com F1-Score de {best_f1:.4f})")
    return best_threshold

def plot_and_save_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    class_names = ['Legítimo', 'Fraude']
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val}', ha='center', va='center', color='white' if val > cm.max() / 1.5 else 'black')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão (Teste, Limiar Otimizado)')
    plt.savefig(save_path)
    plt.close()
    print(f"Matriz de confusão salva em: {save_path}")

# --- 5. Bloco de Execução Principal ---

def main():
    parser = argparse.ArgumentParser(description="Treinamento de GNN para Detecção de Fraude")
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'sage', 'gat'], help='Modelo GNN a ser utilizado.')
    parser.add_argument('--balancing', type=str, default='weights', choices=['none', 'weights', 'oversample'], help='Técnica de balanceamento.')
    # MODIFICAÇÃO: Alterado o caminho padrão para o arquivo GEXF
    parser.add_argument('--data_path', type=str, default='grafo_filtrado_transacoes_c_fraude.gexf', help='Caminho para o arquivo GEXF do grafo.')
    parser.add_argument('--epochs', type=int, default=300, help='Número de épocas.')
    parser.add_argument('--lr', type=float, default=0.005, help='Taxa de aprendizado inicial.')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Canais ocultos.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Taxa de dropout.')
    parser.add_argument('--num_layers', type=int, default=2, help='Número de camadas.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Decaimento de peso (L2).')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = os.path.join("Results_GNN", f"Result_GEXF_{args.model.upper()}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(result_dir, "log_console.txt"))

    print(f"Execução iniciada em: {timestamp}\n\nConfigurações da Rodada:")
    for key, value in vars(args).items():
        print(f"  - {key}: {value}")
    print("-" * 30)

    if not os.path.exists(args.data_path):
        print(f"Erro: Arquivo não encontrado em '{args.data_path}'")
        return
        
    data = load_and_preprocess_data(args.data_path)
    if data is None:
        print("Falha no carregamento dos dados. Abortando.")
        return
        
    data = create_masks(data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsando dispositivo: {device}\n")
    data = data.to(device)

    model_common_args = {'in_channels': data.num_node_features, 'hidden_channels': args.hidden_channels, 'out_channels': 2, 'num_layers': args.num_layers, 'dropout': args.dropout}
    if args.model.lower() == 'gcn':
        model = GCN(**model_common_args).to(device)
    elif args.model.lower() == 'sage':
        model = GraphSAGE(**model_common_args).to(device)
    else:
        gat_hidden = args.hidden_channels // 8 if args.hidden_channels >= 8 else 1
        model = GAT(in_channels=data.num_node_features, hidden_channels=gat_hidden, out_channels=2, heads=8, dropout=args.dropout).to(device)
    
    print("Arquitetura do Modelo:\n", model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=20, min_lr=1e-6)
    
    criterion = None
    if args.balancing == 'weights':
        class_counts = torch.bincount(data.y[data.train_mask])
        if 0 in class_counts:
            print("Aviso: Uma das classes não está presente no conjunto de treino. Balanceamento por pesos desativado.")
            criterion = torch.nn.NLLLoss()
        else:
            class_weights = 1. / class_counts.float()
            criterion = torch.nn.NLLLoss(weight=(class_weights / class_weights.sum()).to(device))
            print("\nUsando balanceamento por pesos de classe na função de perda.")
    else:
        criterion = torch.nn.NLLLoss()
        print("\nBalanceamento por pesos desativado.")
        
    train_indices_original = data.train_mask.nonzero(as_tuple=False).view(-1)
    
    # Lógica de oversampling (se ativada)
    minority_indices = torch.tensor([], device=device)
    if args.balancing == 'oversample':
        print("Usando balanceamento por oversampling da classe minoritária.")
        train_labels = data.y[train_indices_original]
        minority_indices = train_indices_original[train_labels == 1]
        majority_indices = train_indices_original[train_labels == 0]
        if len(minority_indices) == 0 or len(majority_indices) == 0:
            print("Aviso: Nenhuma amostra minoritária ou majoritária no treino. Oversampling desativado.")
            args.balancing = 'none'

    best_val_f1, best_model_state = 0, None
    print("\nIniciando o treinamento...")
    for epoch in range(1, args.epochs + 1):
        current_train_indices = train_indices_original
        if args.balancing == 'oversample' and len(minority_indices) > 0:
            # Realiza undersampling da classe majoritária para igualar a minoritária
            sample_indices = np.random.choice(majority_indices.cpu().numpy(), len(minority_indices), replace=False)
            current_train_indices = torch.cat([minority_indices, torch.tensor(sample_indices, device=device)])

        loss = train(model, data, optimizer, criterion, current_train_indices)
        
        if epoch % 10 == 0:
            val_metrics, _, _ = evaluate(model, data, data.val_mask)
            val_f1 = val_metrics['f1']
            scheduler.step(val_f1)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict()
                print(f'🚀 Época: {epoch:03d}, Perda: {loss:.4f}, Val F1: {val_f1:.4f} (Novo Melhor!)')
            else:
                print(f'   Época: {epoch:03d}, Perda: {loss:.4f}, Val F1: {val_f1:.4f}')

    print("\n✅ Treinamento finalizado!")

    if best_model_state is None:
        best_model_state = model.state_dict()
    model.load_state_dict(best_model_state)
    model_path = os.path.join(result_dir, 'best_model.pth')
    torch.save(best_model_state, model_path)
    print(f"Melhor modelo salvo em '{model_path}'")
    
    best_threshold = find_best_threshold(model, data)
    
    print("\n--- Resultados Finais (Melhor Modelo com Limiar Otimizado) ---")
    for split_name, mask in [("Treino", data.train_mask), ("Validação", data.val_mask), ("Teste", data.test_mask)]:
        metrics, y_true, y_pred = evaluate(model, data, mask, best_threshold)
        print(f"\nMétricas de {split_name}:")
        for key, val in metrics.items():
            print(f"  - {key.capitalize()}: {val:.4f}")
        if split_name == "Teste":
            cm_path = os.path.join(result_dir, "matriz_confusao_teste.png")
            plot_and_save_confusion_matrix(y_true, y_pred, cm_path)

if __name__ == '__main__':
    main()
