import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GraphSAGE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import CONFIG

class SimpleGNN(torch.nn.Module):
    def __init__(self, num_features=32, hidden_dim=CONFIG["hidden_dim"]):
        super().__init__()
        self.sage = GraphSAGE(num_features, hidden_dim, num_layers=2)
        self.lin = torch.nn.Linear(hidden_dim, 1)
    
    def forward(self, x, edge_index):
        x = self.sage(x, edge_index)
        return self.lin(x).squeeze()

def train_gnn(graph_path: str | Path = "data/processed/openpedcan_graph.pt"):
    print("Starting GNN training with validated real response labels...")
    graph = torch.load(graph_path, weights_only=False)
    if "y" not in graph or "train_mask" not in graph:
        raise ValueError(
            "Training requires real graph.y response labels and graph.train_mask; "
            "the current OpenPedCan graph has no drug-response supervision."
        )

    target = graph.y.float().view(-1)
    train_mask = graph.train_mask.bool().view(-1)
    if target.numel() != graph.num_nodes or train_mask.numel() != graph.num_nodes:
        raise ValueError("Real target labels and train_mask must contain one entry per graph node")
    if not train_mask.any() or not torch.isfinite(target[train_mask]).all():
        raise ValueError("Training requires at least one finite real response label")

    model = SimpleGNN(num_features=graph.num_node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    for epoch in range(CONFIG["num_epochs"]):
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index)
        loss = F.mse_loss(out.view(-1)[train_mask], target[train_mask])

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} - Loss: {loss.item():.4f}")

    # Save model
    os.makedirs(CONFIG["artifacts_path"], exist_ok=True)
    model_path = f"{CONFIG['artifacts_path']}model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"✅ GNN model saved to {model_path}")

    # Save simple metrics
    metrics = {
        "final_loss": float(loss.item()),
        "epochs": CONFIG["num_epochs"],
        "training_examples": int(train_mask.sum().item()),
    }
    with open(f"{CONFIG['artifacts_path']}metrics.json", "w") as f:
        json.dump(metrics, f)

    print("🎉 Training completed successfully!")

if __name__ == "__main__":
    train_gnn()