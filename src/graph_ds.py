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

# Adicionar informações de documentos
for _, row in documents.iterrows():
    doc_id = row["document_id"]
    documento = row["documento"]
    
    if doc_id not in doc_features:
        doc_features[doc_id] = {
            "document_id": doc_id,
            "documentos": set(),  # Usamos um conjunto para evitar duplicações
            "documento_tipo": row["documento_tipo"] if pd.notna(row["documento_tipo"]) else "",
            "account_info": [],  # Para armazenar as contas relacionadas
            "level": None  # Inicializa o nível como None
        }
    
    doc_features[doc_id]["documentos"].add(documento)

# Adicionar informações de contas
for _, row in accounts.iterrows():
    doc_id = row["document_id"]
    if doc_id in doc_features:
        doc_features[doc_id]["account_info"].append({
            "account_number": row["account_number"] if pd.notna(row["account_number"]) else "",
            "account_state": row["account_state"] if pd.notna(row["account_state"]) else "",
            "account_city": row["account_city"] if pd.notna(row["account_city"]) else "",
            "account_group_id": row["account_group_id"] if pd.notna(row["account_group_id"]) else "",
            "account_class_id": row["account_class_id"] if pd.notna(row["account_class_id"]) else "",
            "account_email": row["account_dominio_email"] if pd.notna(row["account_dominio_email"]) else ""
        })

# Adicionar informações de levels (garantindo que o nível seja o mais relevante)
for _, row in levels.iterrows():
    doc_id = row["document_id"]
    if doc_id in doc_features:
        if doc_features[doc_id]["level"] is None or row["level"] > doc_features[doc_id]["level"]:
            doc_features[doc_id]["level"] = row["level"]

# Criar grafo direcionado
G = nx.DiGraph()

# Adicionar nós ao grafo com conversão de valores para string
def convert_to_string(value):
    if isinstance(value, (list, pd.Series, set)):  # Verifica se é lista, série ou conjunto
        return ", ".join(map(str, value))  # Converte todos os elementos para string
    if value is None or pd.isna(value):
        return ""
    return str(value)

# Dicionário para mapear documento para document_id
documento_to_doc_id = {}

for doc_id, features in doc_features.items():
    # Convertendo todos os valores para string para evitar erros no GML
    G.add_node(doc_id, **{k: convert_to_string(v) for k, v in features.items()})
    
    # Mapear cada documento para o document_id
    for documento in features["documentos"]:
        documento_to_doc_id[documento] = doc_id

# Criar função para adicionar arestas com conversão para string
def add_transaction_edges(df, src_col, dst_col, trans_id_col, value_col):
    for _, row in df.iterrows():
        src_documento, dst_documento = row[src_col], row[dst_col]
        
        # Verificar se o documento já existe no grafo
        src_doc_id = documento_to_doc_id.get(src_documento, src_documento)
        dst_doc_id = documento_to_doc_id.get(dst_documento, dst_documento)
        
        # Se o documento não estiver mapeado, criar um novo nó
        if src_doc_id not in G:
            G.add_node(src_doc_id, document_id=src_doc_id, documentos=src_documento)
            documento_to_doc_id[src_documento] = src_doc_id
        
        if dst_doc_id not in G:
            G.add_node(dst_doc_id, document_id=dst_doc_id, documentos=dst_documento)
            documento_to_doc_id[dst_documento] = dst_doc_id
        
        trans_id, value = row[trans_id_col], row[value_col]
        
        if not G.has_edge(src_doc_id, dst_doc_id):
            G.add_edge(src_doc_id, dst_doc_id, transaction_ids=[], num_transactions=0, total_value=0.0, avg_value=0.0)
        
        edge = G[src_doc_id][dst_doc_id]
        edge["transaction_ids"].append(convert_to_string(trans_id))
        edge["num_transactions"] += 1
        edge["total_value"] += value
        edge["avg_value"] = edge["total_value"] / edge["num_transactions"]

# Adicionar arestas ao grafo
add_transaction_edges(pix_enviado, "doc_responsavel", "doc_favorecido", "transacao_id", "valor")
add_transaction_edges(ted_enviado, "doc_origem", "doc_destino", "transacao_id", "valor")

# Salvar grafo em GML
nx.write_gml(G, "C:/Users/caleg/Documents/24.2/Monografia/tcc_test/out/graph.gml")