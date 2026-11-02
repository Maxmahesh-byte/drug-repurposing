"""Drug-side knowledge-graph encoder and drug/cell-line pair model.

The response relation is intentionally not loaded. The graph contains only
GDSC drug -> ChEMBL molecule -> direct mechanism target -> gene evidence.
Cell-line expression is encoded separately and fused with the drug embedding
at the pair-scoring head.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv

RELATIONS = (
    ("drug", "exact_name_chembl_mapping", "chembl_molecule"),
    ("chembl_molecule", "chembl_direct_mechanism", "target"),
    ("target", "has_chembl_gene_symbol", "gene"),
)
NODE_TYPES = ("drug", "chembl_molecule", "target", "gene")


@dataclass
class CompactDrugGraph:
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    node_type: torch.Tensor
    drug_indices: torch.Tensor
    drug_ids: list[str]
    num_nodes: int
    num_relations: int
    node_counts: dict[str, int]
    relation_counts: dict[str, int]


def build_compact_drug_graph(graph: Any) -> CompactDrugGraph:
    """Extract the compact, response-free drug-target evidence subgraph.

    All drug nodes are retained. ChEMBL molecules, targets, and genes are
    restricted to nodes used by the three specified evidence relations.
    Each relation is made bidirectional with a distinct reverse-relation ID.
    """
    available_node_types = set(graph.node_types)
    missing_node_types = set(NODE_TYPES) - available_node_types
    if missing_node_types:
        raise ValueError(f"Typed graph is missing node types: {sorted(missing_node_types)}")

    edge_type_lookup = {tuple(edge_type): edge_type for edge_type in graph.edge_types}
    missing_relations = [relation for relation in RELATIONS if relation not in edge_type_lookup]
    if missing_relations:
        raise ValueError(f"Typed graph is missing evidence relations: {missing_relations}")

    node_ids: dict[str, list[str]] = {
        node_type: [str(value) for value in graph[node_type].node_ids]
        for node_type in NODE_TYPES
    }
    if len(node_ids["drug"]) != len(set(node_ids["drug"])):
        raise ValueError("Drug graph contains duplicate drug node IDs.")
    if not node_ids["drug"]:
        raise ValueError("Typed graph contains no drug nodes.")

    raw_edges: dict[tuple[str, str, str], torch.Tensor] = {}
    for relation in RELATIONS:
        edge_index = graph[edge_type_lookup[relation]].edge_index.detach().cpu().long()
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"Invalid edge_index shape for relation {relation}.")
        raw_edges[relation] = edge_index

    # Build only the connected entity subset needed for drug mechanism paths.
    selected: dict[str, set[int]] = {
        "drug": set(range(len(node_ids["drug"]))),
        "chembl_molecule": set(),
        "target": set(),
        "gene": set(),
    }
    drug_to_molecule = raw_edges[RELATIONS[0]]
    molecule_to_target = raw_edges[RELATIONS[1]]
    target_to_gene = raw_edges[RELATIONS[2]]

    selected["chembl_molecule"].update(drug_to_molecule[1].tolist())
    selected["chembl_molecule"].update(molecule_to_target[0].tolist())
    selected["target"].update(molecule_to_target[1].tolist())
    selected["target"].update(target_to_gene[0].tolist())
    selected["gene"].update(target_to_gene[1].tolist())

    selected_ids: dict[str, list[int]] = {
        node_type: sorted(indices) for node_type, indices in selected.items()
    }
    local_index: dict[str, dict[int, int]] = {
        node_type: {old: new for new, old in enumerate(indices)}
        for node_type, indices in selected_ids.items()
    }
    offsets: dict[str, int] = {}
    node_counts: dict[str, int] = {}
    running = 0
    node_type_values: list[int] = []
    for type_index, node_type in enumerate(NODE_TYPES):
        offsets[node_type] = running
        count = len(selected_ids[node_type])
        node_counts[node_type] = count
        node_type_values.extend([type_index] * count)
        running += count

    all_edges: list[torch.Tensor] = []
    all_edge_types: list[torch.Tensor] = []
    relation_counts: dict[str, int] = {}
    for relation_index, relation in enumerate(RELATIONS):
        source_type, _, target_type = relation
        raw = raw_edges[relation]
        kept = [
            (int(source), int(target))
            for source, target in raw.t().tolist()
            if int(source) in local_index[source_type]
            and int(target) in local_index[target_type]
        ]
        relation_counts["::".join(relation)] = len(kept)
        if not kept:
            continue
        source_local = torch.tensor(
            [offsets[source_type] + local_index[source_type][source] for source, _ in kept],
            dtype=torch.long,
        )
        target_local = torch.tensor(
            [offsets[target_type] + local_index[target_type][target] for _, target in kept],
            dtype=torch.long,
        )
        forward = torch.stack((source_local, target_local), dim=0)
        reverse = torch.stack((target_local, source_local), dim=0)
        all_edges.extend((forward, reverse))
        all_edge_types.extend(
            (
                torch.full((len(kept),), relation_index, dtype=torch.long),
                torch.full((len(kept),), relation_index + len(RELATIONS), dtype=torch.long),
            )
        )

    if all_edges:
        edge_index_out = torch.cat(all_edges, dim=1).contiguous()
        edge_type_out = torch.cat(all_edge_types, dim=0).contiguous()
    else:
        edge_index_out = torch.empty((2, 0), dtype=torch.long)
        edge_type_out = torch.empty((0,), dtype=torch.long)

    drug_ids = [node_ids["drug"][index] for index in selected_ids["drug"]]
    drug_indices = torch.arange(node_counts["drug"], dtype=torch.long)
    if edge_index_out.numel() and int(edge_index_out.max()) >= running:
        raise ValueError("Compact drug graph contains an out-of-range node index.")

    return CompactDrugGraph(
        edge_index=edge_index_out,
        edge_type=edge_type_out,
        node_type=torch.tensor(node_type_values, dtype=torch.long),
        drug_indices=drug_indices,
        drug_ids=drug_ids,
        num_nodes=running,
        num_relations=2 * len(RELATIONS),
        node_counts=node_counts,
        relation_counts=relation_counts,
    )


class DrugCellLinePairGNN(nn.Module):
    """Three-layer RGCN drug encoder + expression MLP + pairwise dual-endpoint head.

    Three message-passing layers are required for a gene to traverse the full
    reverse evidence path gene -> target -> ChEMBL molecule -> drug.
    """

    def __init__(
        self,
        num_graph_nodes: int,
        num_node_types: int,
        num_relations: int,
        num_drugs: int,
        num_expression_features: int,
        hidden_dim: int = 64,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        if min(
            num_graph_nodes, num_node_types, num_relations, num_drugs,
            num_expression_features, hidden_dim,
        ) <= 0:
            raise ValueError("All model dimensions must be positive.")
        self.entity_embedding = nn.Embedding(num_graph_nodes, hidden_dim)
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim)
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.node_type_embedding.weight)

        num_bases = min(4, num_relations)
        self.drug_conv1 = RGCNConv(
            hidden_dim, hidden_dim, num_relations=num_relations, num_bases=num_bases
        )
        self.drug_conv2 = RGCNConv(
            hidden_dim, hidden_dim, num_relations=num_relations, num_bases=num_bases
        )
        self.drug_norm1 = nn.LayerNorm(hidden_dim)
        self.drug_norm2 = nn.LayerNorm(hidden_dim)
        self.drug_conv3 = RGCNConv(
            hidden_dim, hidden_dim, num_relations=num_relations, num_bases=num_bases
        )
        self.drug_norm3 = nn.LayerNorm(hidden_dim)
        self.num_message_passing_layers = 3

        self.cell_encoder = nn.Sequential(
            nn.Linear(num_expression_features, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        pair_width = hidden_dim * 4
        self.pair_head = nn.Sequential(
            nn.Linear(pair_width, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.dropout = dropout
        self.num_drugs = num_drugs

    def encode(
        self,
        expression_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_type: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_index = torch.arange(
            self.entity_embedding.num_embeddings,
            device=self.entity_embedding.weight.device,
        )
        graph_x = self.entity_embedding(node_index) + self.node_type_embedding(node_type)
        hidden = self.drug_conv1(graph_x, edge_index, edge_type)
        hidden = self.drug_norm1(F.relu(hidden + graph_x))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden2 = self.drug_conv2(hidden, edge_index, edge_type)
        hidden = self.drug_norm2(F.relu(hidden2 + hidden))
        hidden3 = self.drug_conv3(hidden, edge_index, edge_type)
        hidden = self.drug_norm3(F.relu(hidden3 + hidden))
        return hidden[: self.num_drugs], self.cell_encoder(expression_features)

    def score_pairs(
        self,
        drug_embeddings: torch.Tensor,
        cell_embeddings: torch.Tensor,
        pair_drug_indices: torch.Tensor,
        pair_cell_indices: torch.Tensor,
    ) -> torch.Tensor:
        drug = drug_embeddings[pair_drug_indices]
        cell = cell_embeddings[pair_cell_indices]
        features = torch.cat((drug, cell, drug * cell, torch.abs(drug - cell)), dim=1)
        return self.pair_head(features)
