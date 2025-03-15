import pandas as pd
import networkx as nx

# Caminhos dos arquivos CSV
arquivos_csv = {
    "accounts": "C:/Users/caleg/Documents/24.2/Monografia/tcc_test/data/accounts1.csv",
    "documents": "C:/Users/caleg/Documents/24.2/Monografia/tcc_test/data/documents1.csv",
    "levels": "C:/Users/caleg/Documents/24.2/Monografia/tcc_test/data/levels1.csv",
    "pix_enviado": "C:/Users/caleg/Documents/24.2/Monografia/tcc_test/data/pix_enviado1.csv",
    "ted_enviado": "C:/Users/caleg/Documents/24.2/Monografia/tcc_test/data/ted_envio1.csv"
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
for _, row in documents.iterrows():
    doc_id = row["document_id"]
    doc_features[doc_id] = {
        "document_id": doc_id,
        "documento_tipo": row["documento_tipo"]
    }

# Adicionar informações das contas
for _, row in accounts.iterrows():
    doc_id = row["document_id"]
    if doc_id in doc_features:
        doc_features[doc_id].update({
            "account_state": row["account_state"],
            "account_city": row["account_city"],
            "account_group_id": row["account_group_id"]
        })

# Adicionar informações de levels
for _, row in levels.iterrows():
    doc_id = row["document_id"]
    if doc_id in doc_features:
        doc_features[doc_id]["level"] = row["level"]

# Criar grafo direcionado
G = nx.DiGraph()

# Adicionar nós ao grafo
for doc_id, features in doc_features.items():
    G.add_node(doc_id, **features)

# Criar função para adicionar arestas
def add_transaction_edges(df, src_col, dst_col, trans_id_col, value_col):
    for _, row in df.iterrows():
        src, dst = row[src_col], row[dst_col]
        trans_id, value = row[trans_id_col], row[value_col]
        if not G.has_edge(src, dst):
            G.add_edge(src, dst, transaction_ids=[], num_transactions=0, total_value=0.0, avg_value=0.0)
        edge = G[src][dst]
        edge["transaction_ids"].append(trans_id)
        edge["num_transactions"] += 1
        edge["total_value"] += value
        edge["avg_value"] = edge["total_value"] / edge["num_transactions"]

# Adicionar arestas ao grafo
add_transaction_edges(pix_enviado, "doc_responsavel", "doc_favorecido", "transacao_id", "valor")
add_transaction_edges(ted_enviado, "doc_origem", "doc_destino", "transacao_id", "valor")

# Salvar grafo em GML
nx.write_gml(G, "C:/Users/caleg/Documents/24.2/Monografia/tcc_test/out/graph.gml")
