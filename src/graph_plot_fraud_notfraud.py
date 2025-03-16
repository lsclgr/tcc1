import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

# Caminho do arquivo salvo
graph_path = "/home/luisa-calegari/Documentos/tcc_test/out/graph_final_tcc.gml"

# Carregar o grafo a partir do arquivo GML
G = nx.read_gml(graph_path)

# Verificar o número de nós e arestas
print(f"Número de nós: {G.number_of_nodes()}")
print(f"Número de arestas: {G.number_of_edges()}")

# Filtrar nós com level "Fraude"
fraud_nodes = [node for node, attributes in G.nodes(data=True) if attributes.get("level") == "Fraude"]

# Filtrar nós com outros levels (não "Fraude")
non_fraud_nodes = [node for node, attributes in G.nodes(data=True) if attributes.get("level") == "D"]

# Selecionar 5 nós aleatoriamente com level "Fraude"
selected_fraud_nodes = random.sample(fraud_nodes, min(10, len(fraud_nodes)))

# Selecionar 5 nós aleatoriamente com outros levels
selected_non_fraud_nodes = random.sample(non_fraud_nodes, min(10, len(non_fraud_nodes)))

# Coletar os vizinhos dos nós selecionados
neighbors = set()
for node in selected_fraud_nodes + selected_non_fraud_nodes:
    neighbors.update(G.neighbors(node))

# Criar um subgrafo com os nós selecionados e seus vizinhos
subgraph_nodes = set(selected_fraud_nodes + selected_non_fraud_nodes).union(neighbors)
subgraph = G.subgraph(subgraph_nodes)

# Verificar o número de nós e arestas no subgrafo
print(f"Número de nós no subgrafo: {subgraph.number_of_nodes()}")
print(f"Número de arestas no subgrafo: {subgraph.number_of_edges()}")

# Definir posições fixas para os nós selecionados
pos = {}
num_selected_nodes = len(selected_fraud_nodes) + len(selected_non_fraud_nodes)
angle_step = 2 * np.pi / num_selected_nodes  # Distribuir os nós selecionados em um círculo
radius = 5  # Raio do círculo onde os nós selecionados serão posicionados

# Posicionar os nós de fraude
for i, node in enumerate(selected_fraud_nodes):
    angle = i * angle_step
    pos[node] = np.array([radius * np.cos(angle), radius * np.sin(angle)])  # Posicionar em um círculo

# Posicionar os nós com outros levels
for i, node in enumerate(selected_non_fraud_nodes, start=len(selected_fraud_nodes)):
    angle = i * angle_step
    pos[node] = np.array([radius * np.cos(angle), radius * np.sin(angle)])  # Posicionar em um círculo

# Posicionar os vizinhos próximos aos seus respectivos nós centrais
for node in subgraph.nodes():
    if node not in selected_fraud_nodes and node not in selected_non_fraud_nodes:  # Se for um vizinho
        # Encontrar o nó central ao qual ele está conectado
        connected_central_node = next(
            (n for n in selected_fraud_nodes + selected_non_fraud_nodes 
             if subgraph.has_edge(node, n) or subgraph.has_edge(n, node)), None)
        if connected_central_node:
            # Posicionar o vizinho em uma posição aleatória próxima ao nó central
            pos[node] = pos[connected_central_node] + np.random.uniform(-0.5, 0.5, size=2)

# Plotar o subgrafo
plt.figure(figsize=(12, 8))

# Desenhar os nós de fraude em vermelho
nx.draw_networkx_nodes(subgraph, pos, nodelist=selected_fraud_nodes, node_size=100, node_color="red", label="Fraude")

# Desenhar os nós com outros levels em verde
nx.draw_networkx_nodes(subgraph, pos, nodelist=selected_non_fraud_nodes, node_size=100, node_color="green", label="D")

# Desenhar os vizinhos em azul
nx.draw_networkx_nodes(subgraph, pos, nodelist=neighbors, node_size=50, node_color="skyblue", label="Vizinhos")

# Desenhar as arestas
nx.draw_networkx_edges(subgraph, pos, edge_color="gray")

# Adicionar legenda
plt.legend(scatterpoints=1, frameon=True, title="Legenda")

# Adicionar título
plt.title("Subgrafo com 10 nós de Fraude, 10 level D e seus vizinhos")

# Salvar a imagem
plt.savefig("/home/luisa-calegari/Documentos/tcc_test/img/mixed_subgraph_plot_D.png", dpi=300)

# Mostrar o plot
plt.show()