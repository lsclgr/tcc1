import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random

# Caminhos dos arquivos CSV
arquivos_csv = {
    "accounts": "C:/Users/caleg/OneDrive/Documents/tcc_test/data/accounts1.csv",
    "documents": "C:/Users/caleg/OneDrive/Documents/tcc_test/data/documents1.csv",
    "levels": "C:/Users/caleg/OneDrive/Documents/tcc_test/data/levels1.csv",
    "pix_enviado": "C:/Users/caleg/OneDrive/Documents/tcc_test/data/pix_enviado1.csv",
    "ted_enviado": "C:/Users/caleg/OneDrive/Documents/tcc_test/data/ted_envio1.csv"
}

# Carregar os arquivos CSV em dataframes
dfs = {nome: pd.read_csv(caminho, dtype=str).applymap(lambda x: x.strip() if isinstance(x, str) else x) 
       for nome, caminho in arquivos_csv.items()}

# Criar o grafo
G = nx.Graph()

# Conectar todos os valores dentro de cada linha dos arquivos
for nome, df in dfs.items():
    for _, linha in df.iterrows():
        valores = linha.dropna().astype(str).tolist()  # Remover NaN e garantir strings
        for i in range(len(valores)):
            for j in range(i + 1, len(valores)):
                G.add_edge(valores[i], valores[j])

# Conectar os valores entre arquivos (ligação entre valores em comum)
for index, row in dfs["accounts"].iterrows():
    acc_number = row["account_number"].strip()
    doc_id = row["document_id"].strip()

    # Conectar account_number com outros account_number no pix_enviado
    if "pix_enviado" in dfs and acc_number in dfs["pix_enviado"]["account_number"].values:
        for _, linha_pix in dfs["pix_enviado"].iterrows():
            if linha_pix["account_number"].strip() == acc_number:
                valores_pix = linha_pix.dropna().astype(str).tolist()
                for valor in valores_pix:
                    G.add_edge(acc_number, valor)
                    print(valor)

    # Conectar document_id com documentos e levels
    for dataset in ["documents", "levels"]:
        if dataset in dfs and doc_id in dfs[dataset]["document_id"].values:
            for _, linha in dfs[dataset].iterrows():
                if linha["document_id"].strip() == doc_id:
                    valores_outros = linha.dropna().astype(str).tolist()
                    for valor in valores_outros:
                        G.add_edge(doc_id, valor)
                        
# Exibir informações sobre o grafo
print(f"Número de nós: {G.number_of_nodes()}")
print(f"Número de arestas: {G.number_of_edges()}")

'''
# Plotar uma parte do grafo para evitar lentidão
nó_central = random.choice(list(G.nodes))
vizinhos = list(G.neighbors(nó_central))
G_sub = G.subgraph([nó_central] + vizinhos)

plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G_sub, seed=42)
nx.draw(G_sub, pos, with_labels=True, node_size=800, font_size=10, edge_color="gray", node_color="lightblue")
plt.show()

'''

nx.draw_networkx(G)

'''
G.add_edge('A', 'B', label='Aresta AB')
G.add_edge('B', 'C', label='Aresta BC')
G.add_edge('C', 'A', label='Aresta CA')
'''