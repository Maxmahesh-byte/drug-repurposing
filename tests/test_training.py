import pytest
import torch
from torch_geometric.data import Data

from src.training.train_gnn import train_gnn


def test_training_refuses_graph_without_real_labels(tmp_path):
    graph_path = tmp_path / "unlabeled_graph.pt"
    graph = Data(x=torch.eye(2), edge_index=torch.tensor([[0, 1], [1, 0]]), num_nodes=2)
    torch.save(graph, graph_path)

    with pytest.raises(ValueError, match="real graph.y response labels"):
        train_gnn(graph_path)