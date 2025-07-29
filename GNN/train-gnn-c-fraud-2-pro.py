import argparse
import os
import ast
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

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch.nn import Linear, Dropout

# --- 1. Definição do Modelo GNN (GraphSAGE) ---

class GraphSAGE(torch.nn.Module):
    """
    Modelo GraphSAGE com camadas de convolução, dropout e uma camada de saída linear.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        # Camada de entrada
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        # Camadas ocultas
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        # Camada de saída
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        
        self.dropout = Dropout(p=dropout)
        self.lin_out = Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            # Aplicar ReLU e Dropout em todas as camadas, exceto a última
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        # Camada linear final e softmax para probabilidades
        x = self.lin_out(x)
        return F.log_softmax(x, dim=1)

# --- 2. Funções de Carregamento e Pré-processamento ---

def parse_transactions(trans_str):
    """
    Processa a string 'transacoes' para extrair features agregadas.
    Retorna um dicionário com as novas features.
    """
    if pd.isna(trans_str) or not isinstance(trans_str, str):
        return {
            'trans_total_valor': 0, 'trans_mediana_valor': 0, 
            'trans_std_valor': 0, 'trans_count_pix': 0, 'trans_count_ted': 0
        }

    valores = []
    tipos = {'pix': 0, 'ted': 0, 'pagamento_conta': 0}
    
    # Usar regex para encontrar pares 'valor:tipo'
    transactions = re.findall(r'(\d+\.?\d*):(\w+)', trans_str)
    
    for valor_str, tipo in transactions:
        try:
            valores.append(float(valor_str))
            if tipo in tipos:
                tipos[tipo] += 1
        except (ValueError, TypeError):
            continue # Ignora transações malformadas

    if not valores:
        return {
            'trans_total_valor': 0, 'trans_mediana_valor': 0, 
            'trans_std_valor': 0, 'trans_count_pix': 0, 'trans_count_ted': 0
        }
        
    return {
        'trans_total_valor': np.sum(valores),
        'trans_mediana_valor': np.median(valores),
        'trans_std_valor': np.std(valores),
        'trans_count_pix': tipos.get('pix', 0),
        'trans_count_ted': tipos.get('ted', 0),
    }

def parse_account_info(info_str):
    """
    Processa a string 'account_info' para extrair o número de contas e tipos de e-mail.
    """
    if pd.isna(info_str) or not isinstance(info_str, str):
        return {'num_accounts': 0, 'email_GMAIL': 0, 'email_OTHER': 0}

    # Heurística para contar o número de contas
    num_accounts = info_str.count("'account_number'")
    
    # Contar tipos de email
    gmail_count = info_str.count("'GMAIL'")
    
    return {
        'num_accounts': num_accounts,
        'email_GMAIL': gmail_count,
        'email_OTHER': num_accounts - gmail_count,
    }


def load_and_preprocess_data(gml_path):
    """
    Carrega o grafo GML, extrai e pré-processa os atributos dos nós e arestas.
    """
    print("Carregando o grafo do arquivo GML...")
    try:
        # É importante especificar um `label` para o networkx ler os IDs corretamente
        G = nx.read_gml(gml_path, label='id')
    except Exception as e:
        print(f"Erro ao ler o arquivo GML: {e}")
        return None

    print("Pré-processando os dados dos nós...")
    # Garante uma ordem consistente de nós
    node_ids = sorted(list(G.nodes()))
    node_data = [G.nodes[node_id] for node_id in node_ids]
    
    df_nodes = pd.DataFrame(node_data, index=node_ids)

    # Mapear a variável alvo 'level'
    df_nodes['target'] = df_nodes['level'].apply(lambda x: 1 if x == 'Fraude' else 0)
    
    # --- Engenharia de Features ---
    
    # 1. Processar 'transacoes'
    print("Processando feature 'transacoes' para criar features agregadas...")
    trans_features_df = df_nodes['transacoes'].apply(parse_transactions).apply(pd.Series)
    
    # 2. Processar 'account_info'
    print("Processando feature 'account_info'...")
    account_features_df = df_nodes['account_info'].apply(parse_account_info).apply(pd.Series)

    # 3. Processar 'documento_tipo' (One-Hot Encoding)
    print("Processando feature 'documento_tipo'...")
    doc_tipo_dummies = pd.get_dummies(df_nodes['documento_tipo'], prefix='doc_tipo', dummy_na=True)

    # 4. Selecionar features numéricas diretas
    print("Selecionando features numéricas...")
    numeric_feature_names = [
        'num_transacoes_feitas', 'num_transacoes_recebidas',
        'valor_medio_enviado', 'valor_medio_recebido',
        'desvio_padrao_transacoes', 'num_contas_relacionadas',
        'num_estornos', 'num_cancelamentos', 'num_pagamentos_contas',
        'total_pagamentos_contas'
    ]
    # Filtrar para usar apenas as colunas que existem no dataframe
    available_numeric_features = [col for col in numeric_feature_names if col in df_nodes.columns]
    print(f"Features numéricas encontradas: {available_numeric_features}")

    numeric_df = df_nodes[available_numeric_features].copy()
    # Converter para numérico e preencher NaNs com a média da coluna
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
        numeric_df[col].fillna(numeric_df[col].mean(), inplace=True)
        
    # --- Combinar todas as features ---
    df_final_features = pd.concat([
        numeric_df.reset_index(drop=True),
        trans_features_df.reset_index(drop=True),
        account_features_df.reset_index(drop=True),
        doc_tipo_dummies.reset_index(drop=True)
    ], axis=1)

    # Preencher quaisquer NaNs restantes com 0 (pode ocorrer de dataframes de features)
    df_final_features.fillna(0, inplace=True)
    
    # Escalar todas as features numéricas
    print("Escalando features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_final_features)
    
    # --- Criar tensores para o PyTorch Geometric ---
    x = torch.tensor(features_scaled, dtype=torch.float)
    y = torch.tensor(df_nodes['target'].values, dtype=torch.long)

    # Mapear IDs dos nós para índices inteiros (0 a N-1)
    node_map = {node_id: i for i, node_id in enumerate(node_ids)}
    edges = [(node_map[u], node_map[v]) for u, v in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=len(node_ids))
    
    print("\n--- Resumo do Pré-processamento ---")
    print(f"Dados processados: {data}")
    print(f"Número de features por nó: {data.num_node_features}")
    print(f"Distribuição das classes: {torch.bincount(data.y)}")
    print("------------------------------------\n")
    
    return data


def create_masks(data):
    """
    Cria máscaras de treino, validação e teste de forma estratificada.
    """
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    
    # Divisão estratificada 70% treino, 15% validação, 15% teste
    train_indices, test_val_indices, _, _ = train_test_split(
        indices, data.y.numpy(), test_size=0.3, random_state=42, stratify=data.y.numpy())
    
    val_indices, test_indices, _, _ = train_test_split(
        test_val_indices, data.y.numpy()[test_val_indices], test_size=0.5, random_state=42, stratify=data.y.numpy()[test_val_indices])

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    data.train_mask[train_indices] = True
    data.val_mask[val_indices] = True
    data.test_mask[test_indices] = True
    
    print("Máscaras de dados criadas:")
    print(f"  - Nós de Treino: {data.train_mask.sum().item()}")
    print(f"  - Nós de Validação: {data.val_mask.sum().item()}")
    print(f"  - Nós de Teste: {data.test_mask.sum().item()}")
    
    return data

# --- 3. Funções de Treinamento e Avaliação ---

def train(model, data, optimizer, criterion):
    """
    Executa uma época de treinamento.
    """
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data, split_mask):
    """
    Avalia o modelo em uma máscara de dados específica (treino, val, ou teste).
    """
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    mask = data[split_mask]
    y_true = data.y[mask].cpu()
    y_pred = pred[mask].cpu()
    
    # Probabilidades para a classe positiva (Fraude) para o cálculo do AUC
    y_prob = out[mask][:, 1].exp().cpu()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    }
    return metrics, y_true, y_pred

def plot_and_save_confusion_matrix(y_true, y_pred, save_path):
    """Plota e salva a matriz de confusão."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    # Definir os rótulos dos eixos
    class_names = ['Legítimo', 'Fraude']
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Anotar os valores nas células
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val}', ha='center', va='center', color='red' if val > cm.max()/2. else 'black')
        
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão (Dados de Teste)')
    plt.savefig(save_path)
    plt.close()
    print(f"Matriz de confusão salva em: {save_path}")

# --- 4. Bloco de Execução Principal ---

def main():
    parser = argparse.ArgumentParser(description="Treinamento de GNN para Detecção de Fraude")
    parser.add_argument('--data_path', type=str, default='out/grafo_fraude_c.gml', help='Caminho para o arquivo GML.')
    parser.add_argument('--epochs', type=int, default=200, help='Número de épocas de treinamento.')
    parser.add_argument('--lr', type=float, default=0.01, help='Taxa de aprendizado.')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Número de canais nas camadas ocultas.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Taxa de dropout.')
    parser.add_argument('--num_layers', type=int, default=3, help='Número de camadas SAGEConv.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Decaimento de peso (L2).')
    args = parser.parse_args()

    # --- Carregar e Preparar Dados ---
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

    # --- Inicializar Modelo, Otimizador e Critério de Perda ---
    model = GraphSAGE(
        in_channels=data.num_node_features,
        hidden_channels=args.hidden_channels,
        out_channels=2, # Duas classes: Legítimo (0) e Fraude (1)
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Calcular pesos para lidar com classes desbalanceadas
    class_counts = torch.bincount(data.y)
    class_weights = 1. / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    criterion = torch.nn.NLLLoss(weight=class_weights.to(device))

    # --- Loop de Treinamento e Avaliação ---
    best_val_auc = 0
    best_model_state = None

    print("\nIniciando o treinamento...")
    for epoch in range(1, args.epochs + 1):
        loss = train(model, data, optimizer, criterion)
        
        # Avaliação periódica (a cada 10 épocas)
        if epoch % 10 == 0:
            val_metrics, _, _ = evaluate(model, data, 'val_mask')
            val_auc = val_metrics['roc_auc']

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                # Salvar o estado do modelo com melhor desempenho de validação
                best_model_state = model.state_dict()
                print(f'🚀 Época: {epoch:03d}, Perda: {loss:.4f}, Val AUC: {val_auc:.4f} (Novo Melhor!)')
            else:
                print(f'   Época: {epoch:03d}, Perda: {loss:.4f}, Val AUC: {val_auc:.4f}')

    print("\n✅ Treinamento finalizado!")

    # --- Avaliação Final com o Melhor Modelo ---
    if best_model_state is None:
        print("Nenhum modelo foi salvo. Usando o modelo da última época.")
        best_model_state = model.state_dict()
        
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, 'best_model.pth')
    print("Melhor modelo salvo em 'best_model.pth'")
    
    print("\n--- Resultados Finais (Melhor Modelo) ---")
    train_metrics, _, _ = evaluate(model, data, 'train_mask')
    val_metrics, _, _ = evaluate(model, data, 'val_mask')
    test_metrics, y_true_test, y_pred_test = evaluate(model, data, 'test_mask')

    print("\nMétricas de Treino:")
    for key, val in train_metrics.items(): print(f"  - {key.capitalize()}: {val:.4f}")
    
    print("\nMétricas de Validação:")
    for key, val in val_metrics.items(): print(f"  - {key.capitalize()}: {val:.4f}")

    print("\nMétricas de Teste:")
    for key, val in test_metrics.items(): print(f"  - {key.capitalize()}: {val:.4f}")

    # Plotar e salvar matriz de confusão
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_path = f"matriz_confusao_teste_{timestamp}.png"
    plot_and_save_confusion_matrix(y_true_test, y_pred_test, cm_path)

if __name__ == '__main__':
    main()