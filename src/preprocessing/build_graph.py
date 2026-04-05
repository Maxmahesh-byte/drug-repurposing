import torch
import networkx as nx
import pandas as pd
from torch_geometric.data import Data
from src.utils.config import CONFIG
import os

def build_heterogeneous_graph():
    print("🔨 Building heterogeneous knowledge graph from real GDSC data...")
    os.makedirs(CONFIG["processed_data_path"], exist_ok=True)
    
    # Load real GDSC data
    df = pd.read_csv(f"{CONFIG['raw_data_path']}gdsc2_original.csv")
    
    G = nx.Graph()
    
    # Add nodes
    drugs = df['DRUG_NAME'].unique()
    cell_lines = df['CELL_LINE'].unique()
    
    for drug in drugs:
        G.add_node(drug, type="drug")
    for cell in cell_lines:
        G.add_node(cell, type="cell_line")
    
    # Add edges based on real drug-cell line pairs
    for _, row in df.iterrows():
        G.add_edge(row['DRUG_NAME'], row['CELL_LINE'], 
                  weight=row['IC50'], 
                  auc=row['AUC'])
    
    # Convert to PyTorch Geometric
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G.edges()]).t().contiguous()
    
    # Node features (dummy for now - can be extended with gene expression later)
    x = torch.randn(len(G.nodes()), 32)  # 32-dim features
    
    data = Data(x=x, edge_index=edge_index, num_nodes=len(G.nodes()))
    torch.save(data, f"{CONFIG['processed_data_path']}graph.pt")
    
    print(f"✅ Real GDSC graph built: {data.num_nodes} nodes, {data.num_edges} edges")
    return data

if __name__ == "__main__":
    build_heterogeneous_graph()