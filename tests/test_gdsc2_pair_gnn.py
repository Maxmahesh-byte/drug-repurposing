import torch
from torch_geometric.data import HeteroData

from src.training.gdsc2_pair_gnn import (
    DrugCellLinePairGNN,
    build_compact_drug_graph,
    restrict_compact_drug_graph,
    summarize_drug_evidence_coverage,
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



def test_three_layers_allow_gene_evidence_to_reach_drug_embedding():
    torch.manual_seed(17)
    compact = build_compact_drug_graph(_tiny_graph())
    model = DrugCellLinePairGNN(
        num_graph_nodes=compact.num_nodes,
        num_node_types=4,
        num_relations=compact.num_relations,
        num_drugs=2,
        num_expression_features=8,
        hidden_dim=16,
        dropout=0.0,
    )
    assert model.num_message_passing_layers == 3
    model.eval()
    expression = torch.zeros(1, 8)
    drug_embeddings, _ = model.encode(
        expression, compact.edge_index, compact.edge_type, compact.node_type
    )
    probe = torch.arange(1, 17, dtype=torch.float32)
    (drug_embeddings[0] * probe).sum().backward()

    # Compact node ordering is drug, molecule, target, gene; gene A is node 6.
    gene_a_index = (
        compact.node_counts["drug"]
        + compact.node_counts["chembl_molecule"]
        + compact.node_counts["target"]
    )
    gene_gradient = model.entity_embedding.weight.grad[gene_a_index]
    assert gene_gradient is not None
    assert torch.isfinite(gene_gradient).all()
    assert gene_gradient.abs().sum().item() > 0


def test_evidence_coverage_counts_complete_gene_paths():
    compact = build_compact_drug_graph(_tiny_graph())
    coverage = summarize_drug_evidence_coverage(compact)
    assert coverage["total_graph_drugs"] == 2
    assert coverage["drug_with_chembl_molecule_mapping"]["count"] == 2
    assert coverage["drug_with_direct_target_path"]["count"] == 2
    assert coverage["drug_with_gene_level_path"]["count"] == 2


def test_invalid_typed_relation_indices_are_rejected():
    graph = _tiny_graph()
    graph["drug", "exact_name_chembl_mapping", "chembl_molecule"].edge_index = torch.tensor([[7], [0]])
    try:
        build_compact_drug_graph(graph)
    except ValueError as exc:
        assert "Source node index out of range" in str(exc)
    else:
        raise AssertionError("Out-of-range typed graph edge should be rejected.")



def test_nested_evidence_graph_views_enable_progressively_more_relations():
    compact = build_compact_drug_graph(_tiny_graph())
    mapping = restrict_compact_drug_graph(compact, "chembl_mapping")
    targets = restrict_compact_drug_graph(compact, "target_evidence")
    genes = restrict_compact_drug_graph(compact, "gene_evidence")

    assert mapping.num_nodes == targets.num_nodes == genes.num_nodes == compact.num_nodes
    assert mapping.num_relations == targets.num_relations == genes.num_relations == compact.num_relations
    assert torch.equal(mapping.node_type, targets.node_type)
    assert torch.equal(targets.node_type, genes.node_type)
    assert mapping.edge_type.unique().tolist() == [0, 3]
    assert targets.edge_type.unique().tolist() == [0, 1, 3, 4]
    assert genes.edge_type.unique().tolist() == [0, 1, 2, 3, 4, 5]
    assert mapping.edge_index.shape[1] < targets.edge_index.shape[1] < genes.edge_index.shape[1]


def test_nested_evidence_graph_rejects_unknown_level():
    compact = build_compact_drug_graph(_tiny_graph())
    try:
        restrict_compact_drug_graph(compact, "chemical_structure")
    except ValueError as exc:
        assert "Unknown evidence level" in str(exc)
    else:
        raise AssertionError("Unknown graph evidence level should be rejected.")
