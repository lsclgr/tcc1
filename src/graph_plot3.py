import networkx as nx
import matplotlib.pyplot as plt
import random

# Caminho do arquivo salvo
graph_path = "/home/luisa-calegari/Documentos/tcc_test/out/graph.gml"

# Carregar o grafo a partir do arquivo GML
G = nx.read_gml(graph_path)

# Verificar o número de nós e arestas
print(f"Número de nós: {G.number_of_nodes()}")
print(f"Número de arestas: {G.number_of_edges()}")

# Selecionar 300 nós aleatoriamente
nodes_to_keep = random.sample(list(G.nodes()), 300)

# Criar um subgrafo com os nós selecionados
subgraph = G.subgraph(nodes_to_keep)

# Verificar o número de nós e arestas no subgrafo
print(f"Número de nós no subgrafo: {subgraph.number_of_nodes()}")
print(f"Número de arestas no subgrafo: {subgraph.number_of_edges()}")

# Plotar o subgrafo sem os nomes dos nós
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(subgraph)  # Layout para disposição dos nós
nx.draw(subgraph, pos, with_labels=False, node_size=50, node_color='skyblue', edge_color='gray')
plt.title("Subgrafo com 300 nós")
plt.savefig("/home/luisa-calegari/Documentos/tcc_test/img/subgraph_plot.png", dpi=300)  # Salva a imagem no diretório atual
plt.show()