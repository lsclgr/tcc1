import pandas as pd
import networkx as nx
import numpy as np

# Caminhos dos arquivos CSV
arquivos_csv = {
    "accounts": "/home/luisa-calegari/Documentos/tcc_test/data/accounts1.csv",
    "documents": "/home/luisa-calegari/Documentos/tcc_test/data/documents1.csv",
    "levels": "/home/luisa-calegari/Documentos/tcc_test/data/levels1.csv",
    "pix_enviado": "/home/luisa-calegari/Documentos/tcc_test/data/pix_enviado1.csv",
    "ted_enviado": "/home/luisa-calegari/Documentos/tcc_test/data/ted_envio1.csv"
}

# Carregar os dados
accounts = pd.read_csv(arquivos_csv["accounts"])
documents = pd.read_csv(arquivos_csv["documents"])
levels = pd.read_csv(arquivos_csv["levels"])
pix_enviado = pd.read_csv(arquivos_csv["pix_enviado"])
ted_enviado = pd.read_csv(arquivos_csv["ted_enviado"])

# Filtrar levels apenas com status "finished"
levels = levels[levels["status"] == "finished"]

# Criar dicionário de features por documento
doc_features = {}

# Adicionar informações de documentos
for _, row in documents.iterrows():
    doc_id = row["document_id"]
    documento = row["documento"]
    
    if documento not in doc_features:
        doc_features[documento] = {
            "document_id": doc_id,
            "documento": documento,
            "documento_tipo": row["documento_tipo"] if pd.notna(row["documento_tipo"]) else "",
            "account_info": [],  # Para armazenar as contas relacionadas
            "level": None,  # Inicializa o nível como None
            "num_transacoes_feitas": 0,  # Número de transações feitas
            "num_transacoes_recebidas": 0,  # Número de transações recebidas
            "valor_medio_enviado": 0.0,  # Valor médio enviado
            "valor_medio_recebido": 0.0,  # Valor médio recebido
            "transacoes": [],  # Lista de transações (valor; tipo)
            "desvio_padrao_transacoes": 0.0,  # Desvio padrão do valor das transações
            "num_contas_relacionadas": 0  # Número de contas relacionadas
        }

# Adicionar informações de contas
for _, row in accounts.iterrows():
    doc_id = row["document_id"]
    documento = documents[documents["document_id"] == doc_id]["documento"].values[0]  # Encontrar o documento correspondente
    if documento in doc_features:
        doc_features[documento]["account_info"].append({
            "account_number": row["account_number"] if pd.notna(row["account_number"]) else "",
            "account_state": row["account_state"] if pd.notna(row["account_state"]) else "",
            "account_city": row["account_city"] if pd.notna(row["account_city"]) else "",
            "account_group_id": row["account_group_id"] if pd.notna(row["account_group_id"]) else "",
            "account_class_id": row["account_class_id"] if pd.notna(row["account_class_id"]) else "",
            "account_email": row["account_dominio_email"] if pd.notna(row["account_dominio_email"]) else ""
        })
        doc_features[documento]["num_contas_relacionadas"] += 1  # Incrementa o número de contas relacionadas

# Adicionar informações de levels (garantindo que o nível seja o mais relevante)
for _, row in levels.iterrows():
    doc_id = row["document_id"]
    documento = documents[documents["document_id"] == doc_id]["documento"].values[0]  # Encontrar o documento correspondente
    if documento in doc_features:
        if doc_features[documento]["level"] is None or row["level"] > doc_features[documento]["level"]:
            doc_features[documento]["level"] = row["level"]

# Criar grafo direcionado
G = nx.DiGraph()

# Função para converter valores complexos em strings
def convert_to_string(value):
    if isinstance(value, (list, pd.Series, set)):  # Verifica se é lista, série ou conjunto
        if all(isinstance(item, tuple) for item in value):  # Se for lista de tuplas
            return "; ".join(f"{t[0]}:{t[1]}" for t in value)  # Formata como "id:type; id:type"
        return ", ".join(map(str, value))  # Converte outros tipos para string
    if value is None or pd.isna(value):
        return ""  # Converte None ou NaN para string vazia
    return str(value)

# Função para atualizar as métricas de transações no nó
def update_node_transaction_metrics(node, value, trans_type, is_sender):
    # Atualiza o número de transações feitas ou recebidas
    if is_sender:
        node["num_transacoes_feitas"] += 1
    else:
        node["num_transacoes_recebidas"] += 1

    # Adiciona a transação à lista de transações
    node["transacoes"].append((value, trans_type))

    # Recalcula o valor médio enviado/recebido
    if is_sender:
        total_sent = node["valor_medio_enviado"] * (node["num_transacoes_feitas"] - 1) + value
        node["valor_medio_enviado"] = total_sent / node["num_transacoes_feitas"]
    else:
        total_received = node["valor_medio_recebido"] * (node["num_transacoes_recebidas"] - 1) + value
        node["valor_medio_recebido"] = total_received / node["num_transacoes_recebidas"]

    # Recalcula o desvio padrão das transações
    trans_values = [t[0] for t in node["transacoes"]]
    node["desvio_padrao_transacoes"] = np.std(trans_values) if trans_values else 0.0

# Adicionar nós ao grafo
for documento, features in doc_features.items():
    # Adiciona os nós sem converter valores numéricos para strings
    G.add_node(documento, **features)

# Função para adicionar arestas com atualização das métricas dos nós
def add_transaction_edges(df, src_col, dst_col, trans_id_col, value_col, trans_type):
    for _, row in df.iterrows():
        src_documento, dst_documento = row[src_col], row[dst_col]
        trans_id, value = row[trans_id_col], row[value_col]

        # Verificar se os nós já existem no grafo
        if src_documento not in G:
            G.add_node(src_documento, document_id="", documento=src_documento, documento_tipo="", account_info=[],
                       level=None, num_transacoes_feitas=0, num_transacoes_recebidas=0, valor_medio_enviado=0.0,
                       valor_medio_recebido=0.0, transacoes=[], desvio_padrao_transacoes=0.0, num_contas_relacionadas=0)
        if dst_documento not in G:
            G.add_node(dst_documento, document_id="", documento=dst_documento, documento_tipo="", account_info=[],
                       level=None, num_transacoes_feitas=0, num_transacoes_recebidas=0, valor_medio_enviado=0.0,
                       valor_medio_recebido=0.0, transacoes=[], desvio_padrao_transacoes=0.0, num_contas_relacionadas=0)

        # Atualizar métricas dos nós
        update_node_transaction_metrics(G.nodes[src_documento], value, trans_type, is_sender=True)
        update_node_transaction_metrics(G.nodes[dst_documento], value, trans_type, is_sender=False)

        # Adicionar aresta com valor médio das transações
        if not G.has_edge(src_documento, dst_documento):
            G.add_edge(src_documento, dst_documento, valor_medio_transacoes=0.0, num_transacoes=0, total_value=0.0)

        edge = G[src_documento][dst_documento]
        edge["num_transacoes"] += 1
        edge["total_value"] += value
        edge["valor_medio_transacoes"] = edge["total_value"] / edge["num_transacoes"]

# Adicionar arestas ao grafo
add_transaction_edges(pix_enviado, "doc_responsavel", "doc_favorecido", "transacao_id", "valor", "pix")
add_transaction_edges(ted_enviado, "doc_origem", "doc_destino", "transacao_id", "valor", "ted")

# Antes de salvar, garantir que todos os atributos sejam strings
for u, v, data in G.edges(data=True):
    data["valor_medio_transacoes"] = str(data["valor_medio_transacoes"])

for node, data in G.nodes(data=True):
    for key, value in data.items():
        if isinstance(value, (list, dict, int, float)) or value is None:
            data[key] = convert_to_string(value)

# Salvar grafo em GML
nx.write_gml(G, "/home/luisa-calegari/Documentos/tcc_test/out/graph.gml")

# Verificar o número de nós e arestas
num_nos = G.number_of_nodes()
num_arestas = G.number_of_edges()

print(f"O grafo gerado tem {num_nos} nós e {num_arestas} arestas.")