import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random

# Criar um grafo
G = nx.Graph()

# Lista de arquivos para processar manualmente
arquivos_csv = [
    "C:/Users/caleg/OneDrive\Documents/tcc_test/data/accounts1.csv",
    "C:/Users/caleg/OneDrive\Documents/tcc_test/data/documents1.csv",
    "C:/Users/caleg/OneDrive\Documents/tcc_test/data/levels1.csv",
    "C:/Users/caleg/OneDrive\Documents/tcc_test/data/pix_enviado1.csv"
]

for arquivo in arquivos_csv:
    df = pd.read_csv(arquivo)
    
    for _, linha in df.iterrows():
        valores = linha.dropna().astype(str).tolist()  # Remove NaN e converte para string
        
        # Conectar todos os valores da linha entre si
        for i in range(len(valores)):
            for j in range(i + 1, len(valores)):
                G.add_edge(valores[i], valores[j])

# Exibir informações sobre o grafo
print(f"Número de nós: {G.number_of_nodes()}")
print(f"Número de arestas: {G.number_of_edges()}")

'''
# Plotar o grafo
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)  # Layout do grafo
nx.draw(G, pos, with_labels=True, node_size=500, font_size=8, edge_color="gray")
plt.show()
'''

# Escolher um nó aleatório e pegar seus vizinhos
nó_central = random.choice(list(G.nodes))
vizinhos = list(G.neighbors(nó_central))

# Criar subgrafo com o nó central e seus vizinhos
G_sub = G.subgraph([nó_central] + vizinhos)

# Plotar
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G_sub, seed=42)
nx.draw(G_sub, pos, with_labels=True, node_size=800, font_size=10, edge_color="gray", node_color="lightblue")
plt.show()