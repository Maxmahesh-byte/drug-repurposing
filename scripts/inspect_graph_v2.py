import torch
from pathlib import Path

path = Path("data/processed/graph_v2/typed_graph_v2.pt")

if not path.exists():
    raise FileNotFoundError(path)

data = torch.load(path, map_location="cpu", weights_only=False)

print("GRAPH TYPE:", type(data).__name__)
print("\nNODE STORES")

for node_type in data.node_types:
    store = data[node_type]
    print(f"\n{node_type}")
    print("  nodes:", store.num_nodes)
    print("  attributes:", list(store.keys()))

print("\nEDGE STORES")

for edge_type in data.edge_types:
    store = data[edge_type]
    print(f"\n{edge_type}")
    print("  edges:", store.edge_index.shape[1])
    print("  attributes:", list(store.keys()))

    for key in store.keys():
        value = store[key]
        if torch.is_tensor(value):
            print(f"    {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
