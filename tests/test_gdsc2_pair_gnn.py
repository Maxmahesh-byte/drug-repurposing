import torch
from torch_geometric.data import HeteroData

from src.training.gdsc2_pair_gnn import (
    DrugCellLinePairGNN,
    build_compact_drug_graph,
)


def _tiny_graph():
    graph = HeteroData()
    graph["drug"].num_nodes = 2
    graph["drug"].node_ids = ["GDSC_DRUG:1", "GDSC_DRUG:2"]
    graph["chembl_molecule"].num_nodes = 2
    graph["chembl_molecule"].node_ids = ["CHEMBL:1", "CHEMBL:2"]
    graph["target"].num_nodes = 2
    graph["target"].node_ids = ["TARGET:1", "TARGET:2"]
    graph["gene"].num_nodes = 3
    graph["gene"].node_ids = ["GENE:A", "GENE:B", "GENE:C"]
    graph["cell_line"].num_nodes = 1
    graph["cell_line"].node_ids = ["GDSC_MODEL:SIDM1"]
    graph["drug", "exact_name_chembl_mapping", "chembl_molecule"].edge_index = torch.tensor([[0, 1], [0, 1]])
    graph["chembl_molecule", "chembl_direct_mechanism", "target"].edge_index = torch.tensor([[0, 1], [0, 1]])
    graph["target", "has_chembl_gene_symbol", "gene"].edge_index = torch.tensor([[0, 1], [0, 1]])
    # This relation is deliberately present in the source graph and must not be copied.
    graph["drug", "gdsc2_response", "cell_line"].edge_index = torch.tensor([[0], [0]])
    graph["drug", "gdsc2_response", "cell_line"].ln_ic50 = torch.tensor([123.0])
    return graph


def test_compact_graph_contains_only_target_evidence_relations():
    compact = build_compact_drug_graph(_tiny_graph())
    assert compact.drug_ids == ["GDSC_DRUG:1", "GDSC_DRUG:2"]
    assert compact.node_counts == {
        "drug": 2, "chembl_molecule": 2, "target": 2, "gene": 2
    }
    assert compact.num_relations == 6
    assert compact.edge_index.shape == (2, 12)
    assert compact.edge_type.shape == (12,)
    assert compact.edge_type.max().item() < 6


def test_pair_model_forward_has_two_endpoints():
    compact = build_compact_drug_graph(_tiny_graph())
    model = DrugCellLinePairGNN(
        num_graph_nodes=compact.num_nodes,
        num_node_types=4,
        num_relations=compact.num_relations,
        num_drugs=2,
        num_expression_features=8,
        hidden_dim=16,
    )
    expression = torch.randn(3, 8)
    drug_embeddings, cell_embeddings = model.encode(
        expression, compact.edge_index, compact.edge_type, compact.node_type
    )
    predictions = model.score_pairs(
        drug_embeddings,
        cell_embeddings,
        torch.tensor([0, 1, 0]),
        torch.tensor([0, 1, 2]),
    )
    assert predictions.shape == (3, 2)
    assert torch.isfinite(predictions).all()
