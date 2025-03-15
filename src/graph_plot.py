import networkx as nx
import matplotlib.pyplot as plt

# Caminho do arquivo salvo
graph_path = "C:/Users/caleg/Documents/24.2/Monografia/tcc_test/out/graph.gml"

# Carregar o grafo
G = nx.read_gml(graph_path)

'''
# Plotar o grafo
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)  # Layout para melhor visualização
nx.draw(G, pos, node_size=50, with_labels=False, edge_color="gray", alpha=0.7)
plt.title("Transaction Graph")
plt.show()


'''
# Escolher um nó específico para visualização
#selected_node = list(G.nodes())[0]  # Substitua pelo ID do nó desejado

# Escolher um nó que tenha vizinhos
selected_node = None
for node in G.nodes():
    if len(list(G.neighbors(node))) > 50:
        selected_node = node
        break

if selected_node is None:
    raise ValueError("Nenhum nó com vizinhos encontrado no grafo.")

# Obter os 50 vizinhos mais próximos
top_neighbors = list(nx.single_source_shortest_path_length(G, selected_node, cutoff=50).keys())
subgraph = G.subgraph(top_neighbors)

# Plotar o subgrafo
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(subgraph, seed=42)
nx.draw(subgraph, pos, node_size=50, with_labels=False, edge_color="gray", alpha=0.7)
nx.draw_networkx_nodes(subgraph, pos, nodelist=[selected_node], node_color='red', node_size=100)
plt.title(f"Subgraph of {selected_node} and its 50 nearest neighbors")
plt.show()
