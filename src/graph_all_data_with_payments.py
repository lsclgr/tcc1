import pandas as pd
import networkx as nx
import numpy as np
from glob import glob
import os

# ==============================================
# CONFIGURAÇÕES
# ==============================================
DATA_DIR = "data"  # Pasta onde estão os CSVs
OUTPUT_DIR = "out"  # Pasta de saída
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "grafo_fraude.gml")  # Arquivo de saída

# ==============================================
# FUNÇÕES AUXILIARES
# ==============================================
def load_multiple_csvs(base_name):
    """Carrega e combina múltiplos arquivos CSV com o mesmo prefixo"""
    file_pattern = os.path.join(DATA_DIR, f"{base_name}*.csv")
    files = glob(file_pattern)
    
    if not files:
        print(f"[AVISO] Nenhum arquivo encontrado para: {base_name}*.csv")
        return pd.DataFrame()
    
    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"Carregado: {os.path.basename(file)} ({len(df)} registros)")
        except Exception as e:
            print(f"[ERRO] Falha ao carregar {file}: {str(e)}")
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def convert_to_string(value):
    """Converte valores complexos para strings compatíveis com GML"""
    if isinstance(value, (list, pd.Series, set)):
        if all(isinstance(item, tuple) for item in value):
            return "; ".join(f"{t[0]}:{t[1]}" for t in value)
        return ", ".join(map(str, value))
    if value is None or pd.isna(value):
        return ""
    return str(value)

# ==============================================
# CARREGAMENTO DOS DADOS
# ==============================================
print("\n=== CARREGANDO DADOS ===")
accounts = load_multiple_csvs("accounts")
documents = load_multiple_csvs("documents")
levels = load_multiple_csvs("levels")
pix_enviado = load_multiple_csvs("pix_enviado")
ted_enviado = load_multiple_csvs("ted_envio")
pagamento_conta = load_multiple_csvs("pagamento_conta")

# Verificação básica dos dados essenciais
if accounts.empty or documents.empty:
    raise ValueError("Dados essenciais (accounts/documents) não encontrados ou vazios")

# Filtrar apenas levels com status "finished"
levels = levels[levels["status"] == "finished"] if not levels.empty else pd.DataFrame()

# ==============================================
# PRÉ-PROCESSAMENTO DOS DADOS
# ==============================================
print("\n=== PRÉ-PROCESSAMENTO ===")

# Criar dicionário de features por documento
doc_features = {}

# Processar documentos
for _, row in documents.iterrows():
    doc_id = row["document_id"]
    documento = row["documento"]
    
    if documento not in doc_features:
        doc_features[documento] = {
            "document_id": doc_id,
            "documento": documento,
            "documento_tipo": row.get("documento_tipo", ""),
            "account_info": [],
            "level": None,
            "num_transacoes_feitas": 0,
            "num_transacoes_recebidas": 0,
            "valor_medio_enviado": 0.0,
            "valor_medio_recebido": 0.0,
            "transacoes": [],
            "desvio_padrao_transacoes": 0.0,
            "num_contas_relacionadas": 0,
            "num_estornos": 0,
            "num_cancelamentos": 0,
            "num_pagamentos_contas": 0,
            "total_pagamentos_contas": 0.0
        }

# Processar contas
for _, row in accounts.iterrows():
    doc_id = row["document_id"]
    matching_docs = documents[documents["document_id"] == doc_id]
    
    if not matching_docs.empty:
        documento = matching_docs["documento"].values[0]
        if documento in doc_features:
            doc_features[documento]["account_info"].append({
                "account_number": row.get("account_number", ""),
                "account_state": row.get("account_state", ""),
                "account_city": row.get("account_city", ""),
                "account_group_id": row.get("account_group_id", ""),
                "account_class_id": row.get("account_class_id", ""),
                "account_email": row.get("account_dominio_email", "")
            })
            doc_features[documento]["num_contas_relacionadas"] += 1

# Processar níveis
for _, row in levels.iterrows():
    doc_id = row["document_id"]
    matching_docs = documents[documents["document_id"] == doc_id]
    
    if not matching_docs.empty:
        documento = matching_docs["documento"].values[0]
        if documento in doc_features:
            current_level = doc_features[documento]["level"]
            if current_level is None or row["level"] > current_level:
                doc_features[documento]["level"] = row["level"]

# ==============================================
# CONSTRUÇÃO DO GRAFO
# ==============================================
print("\n=== CONSTRUINDO GRAFO ===")
G = nx.DiGraph()

def update_node_transaction_metrics(node, value, trans_type, is_sender):
    """Atualiza as métricas de transação de um nó"""
    if is_sender:
        node["num_transacoes_feitas"] += 1
        total_sent = node["valor_medio_enviado"] * (node["num_transacoes_feitas"] - 1) + value
        node["valor_medio_enviado"] = total_sent / node["num_transacoes_feitas"]
    else:
        node["num_transacoes_recebidas"] += 1
        total_received = node["valor_medio_recebido"] * (node["num_transacoes_recebidas"] - 1) + value
        node["valor_medio_recebido"] = total_received / node["num_transacoes_recebidas"]
    
    node["transacoes"].append((value, trans_type))
    trans_values = [t[0] for t in node["transacoes"]]
    node["desvio_padrao_transacoes"] = np.std(trans_values) if trans_values else 0.0

def add_transaction_edges(df, src_col, dst_col, trans_id_col, value_col, trans_type):
    """Adiciona transações como arestas ao grafo"""
    if df.empty:
        print(f"[AVISO] DataFrame vazio para transações do tipo: {trans_type}")
        return
    
    for _, row in df.iterrows():
        src_doc = row[src_col]
        dst_doc = row[dst_col]
        trans_id = row[trans_id_col]
        value = row[value_col]

        # Adicionar nós se não existirem
        for doc in [src_doc, dst_doc]:
            if doc not in G:
                G.add_node(doc, **doc_features.get(doc, {
                    "document_id": "",
                    "documento": doc,
                    "documento_tipo": "",
                    "account_info": [],
                    "level": None,
                    "num_transacoes_feitas": 0,
                    "num_transacoes_recebidas": 0,
                    "valor_medio_enviado": 0.0,
                    "valor_medio_recebido": 0.0,
                    "transacoes": [],
                    "desvio_padrao_transacoes": 0.0,
                    "num_contas_relacionadas": 0,
                    "num_estornos": 0,
                    "num_cancelamentos": 0,
                    "num_pagamentos_contas": 0,
                    "total_pagamentos_contas": 0.0
                }))

        # Atualizar métricas
        update_node_transaction_metrics(G.nodes[src_doc], value, trans_type, True)
        update_node_transaction_metrics(G.nodes[dst_doc], value, trans_type, False)

        # Contar estornos/cancelamentos
        if "estorno" in row and row["estorno"] == 1:
            G.nodes[src_doc]["num_estornos"] += 1
        if "cancelamento" in row and row["cancelamento"] == 1:
            G.nodes[src_doc]["num_cancelamentos"] += 1

        # Adicionar/atualizar aresta
        if not G.has_edge(src_doc, dst_doc):
            G.add_edge(src_doc, dst_doc, valor_medio_transacoes=0.0, num_transacoes=0, total_value=0.0)
        
        edge = G[src_doc][dst_doc]
        edge["num_transacoes"] += 1
        edge["total_value"] += value
        edge["valor_medio_transacoes"] = edge["total_value"] / edge["num_transacoes"]

def add_bill_payments(df, src_col, trans_id_col, value_col):
    """Adiciona pagamentos de contas (transações sem destinatário específico)"""
    if df.empty:
        print("[AVISO] DataFrame vazio para pagamentos de conta")
        return
    
    for _, row in df.iterrows():
        src_doc = row[src_col]
        value = row[value_col]

        if src_doc not in G:
            G.add_node(src_doc, **doc_features.get(src_doc, {
                "document_id": "",
                "documento": src_doc,
                "documento_tipo": "",
                "account_info": [],
                "level": None,
                "num_transacoes_feitas": 0,
                "num_transacoes_recebidas": 0,
                "valor_medio_enviado": 0.0,
                "valor_medio_recebido": 0.0,
                "transacoes": [],
                "desvio_padrao_transacoes": 0.0,
                "num_contas_relacionadas": 0,
                "num_estornos": 0,
                "num_cancelamentos": 0,
                "num_pagamentos_contas": 0,
                "total_pagamentos_contas": 0.0
            }))

        # Atualizar métricas
        update_node_transaction_metrics(G.nodes[src_doc], value, "pagamento_conta", True)
        G.nodes[src_doc]["num_pagamentos_contas"] += 1
        G.nodes[src_doc]["total_pagamentos_contas"] += value

        if "estorno" in row and row["estorno"] == 1:
            G.nodes[src_doc]["num_estornos"] += 1
        if "cancelamento" in row and row["cancelamento"] == 1:
            G.nodes[src_doc]["num_cancelamentos"] += 1

# Adicionar todas as transações ao grafo
add_transaction_edges(pix_enviado, "doc_responsavel", "doc_favorecido", "transacao_id", "valor", "pix")
add_transaction_edges(ted_enviado, "doc_origem", "doc_destino", "transacao_id", "valor", "ted")
add_bill_payments(pagamento_conta, "doc_origem", "transacao_id", "valor")

# ==============================================
# PÓS-PROCESSAMENTO E SALVAMENTO
# ==============================================
print("\n=== FINALIZANDO ===")

# Converter atributos para strings (formato GML)
for _, _, data in G.edges(data=True):
    data["valor_medio_transacoes"] = str(data["valor_medio_transacoes"])

for node, data in G.nodes(data=True):
    for key, value in data.items():
        if isinstance(value, (list, dict, int, float)) or value is None:
            data[key] = convert_to_string(value)

# Criar diretório de saída se não existir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Salvar grafo
nx.write_gml(G, OUTPUT_FILE)
print(f"Grafo salvo em: {OUTPUT_FILE}")

# Estatísticas finais
num_nos = G.number_of_nodes()
num_arestas = G.number_of_edges()

print("\n=== RESUMO FINAL ===")
print(f"Total de nós: {num_nos}")
print(f"Total de arestas: {num_arestas}")
print(f"Transações PIX processadas: {len(pix_enviado)}")
print(f"Transações TED processadas: {len(ted_enviado)}")
print(f"Pagamentos de conta processados: {len(pagamento_conta)}")
print("\nProcessamento concluído com sucesso!")