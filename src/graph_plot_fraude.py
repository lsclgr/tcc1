import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

# Caminho do arquivo salvo
graph_path = "/home/luisa-calegari/Documentos/tcc_test/out/graph.gml"

# Carregar o grafo a partir do arquivo GML
G = nx.read_gml(graph_path)

# Verificar o número de nós e arestas
print(f"Número de nós: {G.number_of_nodes()}")
print(f"Número de arestas: {G.number_of_edges()}")

# Filtrar nós com level "Fraude"
fraud_nodes = [node for node, attributes in G.nodes(data=True) if attributes.get("level") == "Fraude"]

# Selecionar 10 nós aleatoriamente com level "Fraude"
selected_fraud_nodes = random.sample(fraud_nodes, min(30, len(fraud_nodes)))

# Coletar os vizinhos dos nós selecionados
neighbors = set()
for node in selected_fraud_nodes:
    neighbors.update(G.neighbors(node))

# Criar um subgrafo com os nós selecionados e seus vizinhos
subgraph_nodes = set(selected_fraud_nodes).union(neighbors)
subgraph = G.subgraph(subgraph_nodes)

# Verificar o número de nós e arestas no subgrafo
print(f"Número de nós no subgrafo: {subgraph.number_of_nodes()}")
print(f"Número de arestas no subgrafo: {subgraph.number_of_edges()}")

# Definir posições fixas para os nós de fraude
pos = {}
num_fraud_nodes = len(selected_fraud_nodes)
angle_step = 2 * np.pi / num_fraud_nodes  # Distribuir os nós de fraude em um círculo
radius = 5  # Raio do círculo onde os nós de fraude serão posicionados

for i, node in enumerate(selected_fraud_nodes):
    angle = i * angle_step
    pos[node] = np.array([radius * np.cos(angle), radius * np.sin(angle)])  # Posicionar em um círculo

# Posicionar os vizinhos próximos aos seus respectivos nós de fraude
for node in subgraph.nodes():
    if node not in selected_fraud_nodes:  # Se for um vizinho
        # Encontrar o nó de fraude ao qual ele está conectado
        connected_fraud_node = next((n for n in selected_fraud_nodes if subgraph.has_edge(node, n) or subgraph.has_edge(n, node)), None)
        if connected_fraud_node:
            # Posicionar o vizinho em uma posição aleatória próxima ao nó de fraude
            pos[node] = pos[connected_fraud_node] + np.random.uniform(-0.5, 0.5, size=2)

# Plotar o subgrafo
plt.figure(figsize=(12, 8))
nx.draw(subgraph, pos, with_labels=False, node_size=50, node_color="skyblue", edge_color='gray')

# Destacar os nós de fraude
nx.draw_networkx_nodes(subgraph, pos, nodelist=selected_fraud_nodes, node_size=100, node_color="red")

# Adicionar título
plt.title("Subgrafo com 10 nós de Fraude (separados) e seus vizinhos (próximos)")

# Salvar a imagem
plt.savefig("/home/luisa-calegari/Documentos/tcc_test/img/fraud_subgraph_plot_separated_close_neighbors2.png", dpi=300)

# Mostrar o plot
plt.show()