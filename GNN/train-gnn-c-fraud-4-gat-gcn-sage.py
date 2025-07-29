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

# Importar as camadas GNN necessárias
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv, GCNConv
from torch.nn import Linear, Dropout

# --- 1. Definição dos Modelos ---

class GCN(torch.nn.Module):
    """
    Modelo Graph Convolutional Network (GCN).
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = Dropout(p=dropout)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return F.log_softmax(x, dim=1)

class GraphSAGE(torch.nn.Module):
    """
    Modelo GraphSAGE com camadas de convolução, dropout e uma camada de saída linear.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = Dropout(p=dropout)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    """
    Modelo Graph Attention Network (GAT).
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# --- 2. Gerenciador de Logs ---

class Logger(object):
    """
    Classe para redirecionar a saída do print para o console e um arquivo simultaneamente.
    """
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# --- 3. Funções de Carregamento e Pré-processamento ---

def parse_transactions(trans_str):
    if pd.isna(trans_str) or not isinstance(trans_str, str):
        return {'trans_total_valor': 0, 'trans_mediana_valor': 0, 'trans_std_valor': 0, 'trans_count_pix': 0, 'trans_count_ted': 0}
    
    valores, tipos = [], {'pix': 0, 'ted': 0, 'pagamento_conta': 0}
    transactions = re.findall(r'(\d+\.?\d*):(\w+)', trans_str)
    
    for valor_str, tipo in transactions:
        try:
            valores.append(float(valor_str))
            if tipo in tipos: tipos[tipo] += 1
        except (ValueError, TypeError): continue

    if not valores:
        return {'trans_total_valor': 0, 'trans_mediana_valor': 0, 'trans_std_valor': 0, 'trans_count_pix': 0, 'trans_count_ted': 0}
    
    return {'trans_total_valor': np.sum(valores), 'trans_mediana_valor': np.median(valores), 
            'trans_std_valor': np.std(valores), 'trans_count_pix': tipos.get('pix', 0), 'trans_count_ted': tipos.get('ted', 0)}

def parse_account_info(info_str):
    if pd.isna(info_str) or not isinstance(info_str, str):
        return {'num_accounts': 0, 'email_GMAIL': 0, 'email_OTHER': 0}
    
    num_accounts = info_str.count("'account_number'")
    gmail_count = info_str.count("'GMAIL'")
    return {'num_accounts': num_accounts, 'email_GMAIL': gmail_count, 'email_OTHER': num_accounts - gmail_count}

def load_and_preprocess_data(gml_path):
    print("Carregando o grafo do arquivo GML...")
    try:
        G = nx.read_gml(gml_path, label='id')
    except Exception as e:
        print(f"Erro ao ler o arquivo GML: {e}"); return None

    print("Pré-processando os dados dos nós...")
    node_ids = sorted(list(G.nodes()))
    df_nodes = pd.DataFrame([G.nodes[node_id] for node_id in node_ids], index=node_ids)
    df_nodes['target'] = df_nodes['level'].apply(lambda x: 1 if x == 'Fraude' else 0)
    
    print("Processando features...")
    trans_features_df = df_nodes['transacoes'].apply(parse_transactions).apply(pd.Series)
    account_features_df = df_nodes['account_info'].apply(parse_account_info).apply(pd.Series)
    doc_tipo_dummies = pd.get_dummies(df_nodes['documento_tipo'], prefix='doc_tipo', dummy_na=True)
    
    numeric_feature_names = ['num_transacoes_feitas', 'num_transacoes_recebidas', 'valor_medio_enviado', 'valor_medio_recebido', 'desvio_padrao_transacoes', 'num_contas_relacionadas', 'num_estornos', 'num_cancelamentos', 'num_pagamentos_contas', 'total_pagamentos_contas']
    available_numeric_features = [col for col in numeric_feature_names if col in df_nodes.columns]
    print(f"Features numéricas encontradas: {available_numeric_features}")
    numeric_df = df_nodes[available_numeric_features].copy()
    
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
        numeric_df[col].fillna(numeric_df[col].mean(), inplace=True)
        
    df_final_features = pd.concat([numeric_df.reset_index(drop=True), trans_features_df.reset_index(drop=True), account_features_df.reset_index(drop=True), doc_tipo_dummies.reset_index(drop=True)], axis=1)
    df_final_features.fillna(0, inplace=True)
    
    print("Escalando features...")
    features_scaled = StandardScaler().fit_transform(df_final_features)
    
    x = torch.tensor(features_scaled, dtype=torch.float)
    y = torch.tensor(df_nodes['target'].values, dtype=torch.long)
    
    node_map = {node_id: i for i, node_id in enumerate(node_ids)}
    edges = [(node_map[u], node_map[v]) for u, v in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=len(node_ids))
    
    print("\n--- Resumo do Pré-processamento ---")
    print(f"Dados processados: {data}\nNúmero de features por nó: {data.num_node_features}\nDistribuição das classes: {torch.bincount(data.y)}")
    print("------------------------------------\n")
    return data

def create_masks(data):
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    train_indices, test_val_indices, _, _ = train_test_split(indices, data.y.numpy(), test_size=0.3, random_state=42, stratify=data.y.numpy())
    val_indices, test_indices, _, _ = train_test_split(test_val_indices, data.y.numpy()[test_val_indices], test_size=0.5, random_state=42, stratify=data.y.numpy()[test_val_indices])
    
    for split, idx in {'train_mask': train_indices, 'val_mask': val_indices, 'test_mask': test_indices}.items():
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = True
        data[split] = mask
        
    print(f"Máscaras de dados criadas:\n  - Nós de Treino: {data.train_mask.sum().item()}\n  - Nós de Validação: {data.val_mask.sum().item()}\n  - Nós de Teste: {data.test_mask.sum().item()}")
    return data

# --- 4. Funções de Treinamento e Avaliação ---

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
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
    
    metrics = {'accuracy': accuracy_score(y_true, y_pred), 'precision': precision_score(y_true, y_pred, zero_division=0), 'recall': recall_score(y_true, y_pred, zero_division=0), 'f1': f1_score(y_true, y_pred, zero_division=0), 'roc_auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5}
    return metrics, y_true, y_pred

def find_best_threshold(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = out.exp()

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
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    class_names = ['Legítimo', 'Fraude']
    ax.set_xticks(np.arange(len(class_names))); ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val}', ha='center', va='center', color='white' if val > cm.max()/1.5 else 'black')
        
    plt.xlabel('Predito'); plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão (Teste, Limiar Otimizado)')
    plt.savefig(save_path); plt.close()
    print(f"Matriz de confusão salva em: {save_path}")

# --- 5. Bloco de Execução Principal ---

def main():
    parser = argparse.ArgumentParser(description="Treinamento de GNN para Detecção de Fraude")
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'sage', 'gat'], help='Modelo a ser utilizado (gcn, sage ou gat).')
    parser.add_argument('--data_path', type=str, default='out/grafo_fraude_c.gml', help='Caminho para o arquivo GML.')
    parser.add_argument('--epochs', type=int, default=500, help='Número de épocas de treinamento.')
    parser.add_argument('--lr', type=float, default=0.01, help='Taxa de aprendizado.')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Número de canais nas camadas ocultas.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Taxa de dropout.')
    parser.add_argument('--num_layers', type=int, default=3, help='Número de camadas convolucionais.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Decaimento de peso (L2).')
    args = parser.parse_args()

    # --- Configuração de Pastas e Logs ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = os.path.join("Results_GNN", f"Result_{args.model.upper()}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    log_file_path = os.path.join(result_dir, "log_console.txt")
    sys.stdout = Logger(log_file_path)

    print(f"Execução iniciada em: {timestamp}")
    print("\nConfigurações da Rodada:")
    for key, value in vars(args).items():
        print(f"  - {key}: {value}")
    print("-" * 30)

    # --- Carregar e Preparar Dados ---
    if not os.path.exists(args.data_path):
        print(f"Erro: Arquivo não encontrado em '{args.data_path}'"); return
        
    data = load_and_preprocess_data(args.data_path)
    if data is None: print("Falha no carregamento dos dados. Abortando."); return
        
    data = create_masks(data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsando dispositivo: {device}\n")
    data = data.to(device)

    # --- Inicializar Modelo, Otimizador e Critério de Perda ---
    model_args = {'in_channels': data.num_node_features, 'hidden_channels': args.hidden_channels, 'out_channels': 2, 'num_layers': args.num_layers, 'dropout': args.dropout}
    
    if args.model.lower() == 'gcn':
        model = GCN(**model_args).to(device)
    elif args.model.lower() == 'sage':
        model = GraphSAGE(**model_args).to(device)
    elif args.model.lower() == 'gat':
        gat_hidden = args.hidden_channels // 8 if args.hidden_channels >= 8 else 1
        model = GAT(in_channels=data.num_node_features, hidden_channels=gat_hidden, out_channels=2, heads=8, dropout=args.dropout).to(device)
    else:
        print(f"Modelo '{args.model}' não reconhecido."); return
    
    print("Arquitetura do Modelo:")
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_counts = torch.bincount(data.y); class_weights = 1. / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    criterion = torch.nn.NLLLoss(weight=class_weights.to(device))

    # --- Loop de Treinamento ---
    best_val_f1 = 0
    best_model_state = None
    print("\nIniciando o treinamento...")
    for epoch in range(1, args.epochs + 1):
        loss = train(model, data, optimizer, criterion)
        
        if epoch % 10 == 0:
            val_metrics, _, _ = evaluate(model, data, data.val_mask)
            val_f1 = val_metrics['f1']
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict()
                print(f'🚀 Época: {epoch:03d}, Perda: {loss:.4f}, Val F1: {val_f1:.4f} (Novo Melhor!)')
            else:
                print(f'   Época: {epoch:03d}, Perda: {loss:.4f}, Val F1: {val_f1:.4f}')

    print("\n✅ Treinamento finalizado!")

    # --- Avaliação Final com Melhor Modelo e Limiar Otimizado ---
    if best_model_state is None:
        print("Nenhum modelo foi salvo. Usando o modelo da última época.")
        best_model_state = model.state_dict()
        
    model.load_state_dict(best_model_state)
    model_path = os.path.join(result_dir, 'best_model.pth')
    torch.save(best_model_state, model_path)
    print(f"Melhor modelo salvo em '{model_path}'")
    
    best_threshold = find_best_threshold(model, data)
    
    print("\n--- Resultados Finais (Melhor Modelo com Limiar Otimizado) ---")
    train_metrics, _, _ = evaluate(model, data, data.train_mask, best_threshold)
    val_metrics, _, _ = evaluate(model, data, data.val_mask, best_threshold)
    test_metrics, y_true_test, y_pred_test = evaluate(model, data, data.test_mask, best_threshold)

    print("\nMétricas de Treino:")
    for key, val in train_metrics.items(): print(f"  - {key.capitalize()}: {val:.4f}")
    
    print("\nMétricas de Validação:")
    for key, val in val_metrics.items(): print(f"  - {key.capitalize()}: {val:.4f}")

    print("\nMétricas de Teste:")
    for key, val in test_metrics.items(): print(f"  - {key.capitalize()}: {val:.4f}")

    cm_path = os.path.join(result_dir, "matriz_confusao_teste.png")
    plot_and_save_confusion_matrix(y_true_test, y_pred_test, cm_path)

if __name__ == '__main__':
    main()