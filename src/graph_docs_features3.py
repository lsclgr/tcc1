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
dfs = {key: pd.read_csv(path) for key, path in arquivos_csv.items()}

# Filtrar levels apenas com status "finished"
dfs["levels"] = dfs["levels"][dfs["levels"]["status"] == "finished"]

# Criar dicionário para armazenar nós únicos
node_map = {}

def get_or_create_node(document_id, documento):
    key = (document_id, documento)
    if key not in node_map:
        node_map[key] = {
            "document_id": document_id,
            "documento": documento,
            "account_info": [],
            "level": ""
        }
    return key

# Adicionar informações de documentos
for _, row in dfs["documents"].iterrows():
    get_or_create_node(row["document_id"], row["documento"])

# Adicionar informações de contas
for _, row in dfs["accounts"].iterrows():
    key = get_or_create_node(row["document_id"], "Unknown")
    node_map[key]["account_info"].append({
        "account_number": row["account_number"],
        "account_email": row["account_dominio_email"]
    })

# Adicionar informações de levels
for _, row in dfs["levels"].iterrows():
    key = get_or_create_node(row["document_id"], "Unknown")
    node_map[key]["level"] = row["level"]

# Criar grafo
G = nx.DiGraph()
for key, attributes in node_map.items():
    G.add_node(key, **attributes)

def add_transaction_edges(df, src_col, dst_col, trans_id_col, value_col):
    for _, row in df.iterrows():
        src_key = get_or_create_node(row[src_col], "Unknown")
        dst_key = get_or_create_node(row[dst_col], "Unknown")
        if not G.has_edge(src_key, dst_key):
            G.add_edge(src_key, dst_key, transaction_ids=[], num_transactions=0, total_value=0.0, avg_value=0.0)
        edge = G[src_key][dst_key]
        edge["transaction_ids"].append(row[trans_id_col])
        edge["num_transactions"] += 1
        edge["total_value"] += row[value_col]
        edge["avg_value"] = edge["total_value"] / edge["num_transactions"]

# Escolher qual tipo de transação incluir
usar_pix = True
usar_ted = True
if usar_pix:
    add_transaction_edges(dfs["pix_enviado"], "doc_responsavel", "doc_favorecido", "transacao_id", "valor")
if usar_ted:
    add_transaction_edges(dfs["ted_enviado"], "doc_origem", "doc_destino", "transacao_id", "valor")

# Salvar grafo em GML
nx.write_gml(G, "C:/Users/caleg/Documents/24.2/Monografia/tcc_test/out/graph.gml")
