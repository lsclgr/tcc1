import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

# Caminho do arquivo salvo
graph_path = "C:/Users/caleg/Documents/24.2/Monografia/tcc_test/out/graph.gml"

# Carregar o grafo
G = nx.read_gml(graph_path)

# Salvar o grafo em HTML usando pyvis
net = Network(notebook=True, height="750px", width="100%")
net.from_nx(G)
html_output_path = "C:/Users/caleg/Documents/24.2/Monografia/tcc_test/out/graph.html"
net.show(html_output_path)

# Salvar o grafo em PNG usando matplotlib
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)  # Layout para melhor visualização
nx.draw(G, pos, node_size=50, with_labels=False, edge_color="gray", alpha=0.7)
plt.title("Transaction Graph")
png_output_path = "C:/Users/caleg/Documents/24.2/Monografia/tcc_test/out/graph.png"
plt.savefig(png_output_path, dpi=300, bbox_inches="tight")  # Salva em PNG
plt.close()  # Fecha a figura para liberar memória