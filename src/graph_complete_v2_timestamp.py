import pandas as pd
import numpy as np
import os
from itertools import combinations

# ==============================================
# CONFIGURAÇÕES
# ==============================================
# O arquivo CSV de entrada deve estar no mesmo diretório que o script
DATA_FILE = "nos.csv"
OUTPUT_DIR = "out"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "grafo_v2.gml")

# ==============================================
# FUNÇÕES AUXILIARES
# ==============================================
def convert_to_string(value):
    """
    Converte qualquer valor para uma representação em string segura para o formato GML.
    """
    if value is None or pd.isna(value):
        return ""
    # Remove aspas para evitar problemas na formatação do GML
    return str(value).replace('"', "'")

def write_gml_manual(nodes, edges, filename):
    """
    Gera manualmente um arquivo no formato GML a partir de listas de nós e arestas.
    
    Args:
        nodes (list): Uma lista de dicionários, onde cada dicionário representa um nó e seus atributos.
        edges (list): Uma lista de dicionários, onde cada dicionário representa uma aresta.
        filename (str): O caminho do arquivo de saída.
    """
    print("Iniciando a geração manual do arquivo GML...")
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("graph [\n")
            f.write("  directed 1\n")

            # Escreve os nós
            for node_attrs in nodes:
                node_id = convert_to_string(node_attrs.get("documento"))
                if not node_id:
                    continue  # Pula nós sem um 'documento' válido

                f.write("  node [\n")
                f.write(f'    id "{node_id}"\n')
                for key, value in node_attrs.items():
                    # O 'id' já foi escrito, então pulamos o atributo 'documento'
                    if key == 'documento':
                        continue
                    # Escreve todos os outros atributos
                    f.write(f'    {key} "{convert_to_string(value)}"\n')
                f.write("  ]\n")

            # Escreve as arestas
            for edge_attrs in edges:
                source_id = convert_to_string(edge_attrs.get("source"))
                target_id = convert_to_string(edge_attrs.get("target"))

                if not source_id or not target_id:
                    continue # Pula arestas inválidas

                f.write("  edge [\n")
                f.write(f'    source "{source_id}"\n')
                f.write(f'    target "{target_id}"\n')
                # Adiciona outros atributos da aresta, se houver
                for key, value in edge_attrs.items():
                    if key not in ["source", "target"]:
                        f.write(f'    {key} "{convert_to_string(value)}"\n')
                f.write("  ]\n")

            f.write("]\n")
        print(f"Arquivo GML gerado com sucesso em: {filename}")
    except Exception as e:
        print(f"[ERRO] Falha ao escrever o arquivo GML: {str(e)}")


# ==============================================
# CARREGAMENTO DOS DADOS
# ==============================================
print("\n=== CARREGANDO DADOS ===")
if not os.path.exists(DATA_FILE):
    print(f"[ERRO] Arquivo de dados não encontrado: {DATA_FILE}")
else:
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"Carregado: {DATA_FILE} ({len(df)} registros)")

        # ==============================================
        # PREPARAÇÃO PARA O GRAFO
        # ==============================================
        print("\n=== PREPARANDO DADOS PARA O GRAFO ===")
        
        # Converte o DataFrame em uma lista de dicionários para os nós
        # Garante que todos os valores sejam tipos nativos do Python
        df = df.replace({np.nan: None})
        nodes_list = df.to_dict(orient='records')
        edges_list = []

        print(f"Processando {len(nodes_list)} nós.")

        # Agrupa documentos por 'corporation_id' para criar as arestas
        corporation_groups = df.dropna(subset=['corporation_id']).groupby('corporation_id')['documento'].apply(list)

        edge_count = 0
        for corp_id, documents in corporation_groups.items():
            if len(documents) > 1:
                # Cria arestas (bidirecionais) entre todos os pares de documentos na mesma corporação
                for doc1, doc2 in combinations(documents, 2):
                    edges_list.append({"source": doc1, "target": doc2, "reason": "shared_corporation"})
                    edges_list.append({"source": doc2, "target": doc1, "reason": "shared_corporation"})
                    edge_count += 2
        
        print(f"Identificadas {edge_count} arestas baseadas em 'corporation_id' compartilhado.")

        # ==============================================
        # SALVAMENTO DO GRAFO
        # ==============================================
        print("\n=== SALVANDO O GRAFO ===")
        
        # Cria o diretório de saída se ele não existir
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Chama a função para escrever o arquivo GML
        write_gml_manual(nodes_list, edges_list, OUTPUT_FILE)
        
        # ==============================================
        # ESTATÍSTICAS FINAIS
        # ==============================================
        print("\n=== RESUMO FINAL ===")
        print(f"Total de nós processados: {len(nodes_list)}")
        print(f"Total de arestas geradas: {len(edges_list)}")
        print("\nProcessamento concluído com sucesso!")

    except Exception as e:
        print(f"[ERRO GERAL] Ocorreu um erro durante o processamento: {str(e)}")