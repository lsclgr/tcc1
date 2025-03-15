import networkx as nx
import matplotlib.pyplot as plt

# Caminho do arquivo salvo
graph_path = "C:/Users/caleg/Documents/24.2/Monografia/tcc_test/out/graph.gml"

# Carregar o grafo
G = nx.read_gml(graph_path)

# Encontrar um nó que esteja enviando e recebendo conexões
selected_node = None
for node in G.nodes():
    if len(list(G.predecessors(node))) > 0 and len(list(G.successors(node))) > 0:
        selected_node = node
        break

if selected_node is None:
    raise ValueError("Nenhum nó encontrado com conexões de entrada e saída.")

# Obter vizinhos e vizinhos dos vizinhos
neighbors = set(G.predecessors(selected_node)) | set(G.successors(selected_node))
expanded_neighbors = set(neighbors)
for neighbor in neighbors:
    expanded_neighbors.update(G.predecessors(neighbor))
    expanded_neighbors.update(G.successors(neighbor))

# Garantir que tenha no mínimo 100 nós
while len(expanded_neighbors) < 100:
    additional_nodes = set()
    for node in expanded_neighbors:
        additional_nodes.update(G.predecessors(node))
        additional_nodes.update(G.successors(node))
    if not additional_nodes:
        break
    expanded_neighbors.update(additional_nodes)

# Criar subgrafo
subgraph = G.subgraph(expanded_neighbors)

# Plotar o subgrafo
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(subgraph, seed=42)
nx.draw(subgraph, pos, node_size=50, with_labels=False, edge_color="gray", alpha=0.7)
#nx.draw_networkx_nodes(subgraph, pos, nodelist=[selected_node], node_size=100)
plt.title("Graph segment with at least 100 nodes")
plt.show()
