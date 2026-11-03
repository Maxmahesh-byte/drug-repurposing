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
OPENPEDCAN_RELATIONS = (
    ("sample", "has_gene_variant", "gene"),
    ("tumor_type", "copy_number_in_tumor", "gene"),
    ("sample", "classified_as", "tumor_type"),
    ("sample", "has_diagnosis", "diagnosis"),
    ("sample", "from_cohort", "cohort"),
    ("sample", "has_molecular_subtype", "molecular_subtype"),
    ("sample", "has_fusion", "fusion"),
    ("fusion", "fusion_gene", "gene"),
)
NODE_TYPES = ("drug", "chembl_molecule", "target", "gene")
OPENPEDCAN_NODE_TYPES = (
    "sample", "tumor_type", "diagnosis", "cohort", "molecular_subtype", "fusion"
)


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
    node_types: tuple[str, ...]
    relation_names: tuple[tuple[str, str, str], ...]


def build_compact_drug_graph(graph: Any) -> CompactDrugGraph:
    """Extract drug evidence plus pediatric OpenPedCan context without response edges.

    OpenPedCan messages are restricted to genes already linked to the drug-target
    subgraph. This keeps the graph compact while allowing mutation, fusion,
    tumour-level copy-number, and sample metadata evidence to shape shared gene
    representations. The original GDSC2 response relation is never copied.
    """
    available_node_types = set(graph.node_types)
    missing_node_types = set(NODE_TYPES) - available_node_types
    if missing_node_types:
        raise ValueError(f"Typed graph is missing node types: {sorted(missing_node_types)}")

    edge_type_lookup = {tuple(edge_type): edge_type for edge_type in graph.edge_types}
    missing_relations = [relation for relation in RELATIONS if relation not in edge_type_lookup]
    if missing_relations:
        raise ValueError(f"Typed graph is missing evidence relations: {missing_relations}")

    node_types = tuple(
        [*NODE_TYPES, *[name for name in OPENPEDCAN_NODE_TYPES if name in available_node_types]]
    )
    all_relations = (*RELATIONS, *OPENPEDCAN_RELATIONS)
    relation_names = tuple(
        relation for relation in all_relations
        if relation in edge_type_lookup
        and relation[0] in node_types
        and relation[2] in node_types
    )

    node_ids: dict[str, list[str]] = {
        node_type: [str(value) for value in graph[node_type].node_ids]
        for node_type in node_types
    }
    for node_type, values in node_ids.items():
        if len(values) != len(set(values)):
            raise ValueError(f"{node_type} graph contains duplicate node IDs.")
    if not node_ids["drug"]:
        raise ValueError("Typed graph contains no drug nodes.")

    raw_edges: dict[tuple[str, str, str], torch.Tensor] = {}
    for relation in relation_names:
        edge_index = graph[edge_type_lookup[relation]].edge_index.detach().cpu().long()
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"Invalid edge_index shape for relation {relation}.")
        source_type, _, target_type = relation
        if edge_index.numel():
            if int(edge_index.min()) < 0:
                raise ValueError(f"Negative node index in relation {relation}.")
            if int(edge_index[0].max()) >= len(node_ids[source_type]):
                raise ValueError(f"Source node index out of range in relation {relation}.")
            if int(edge_index[1].max()) >= len(node_ids[target_type]):
                raise ValueError(f"Target node index out of range in relation {relation}.")
        raw_edges[relation] = edge_index

    # Seed the compact graph from the GDSC2 -> ChEMBL -> target -> gene path.
    selected: dict[str, set[int]] = {node_type: set() for node_type in node_types}
    selected["drug"] = set(range(len(node_ids["drug"])))
    for relation_index, relation in enumerate(RELATIONS):
        edge = raw_edges[relation]
        if relation_index == 0:
            selected["chembl_molecule"].update(edge[1].tolist())
        elif relation_index == 1:
            selected["chembl_molecule"].update(edge[0].tolist())
            selected["target"].update(edge[1].tolist())
        elif relation_index == 2:
            selected["target"].update(edge[0].tolist())
            selected["gene"].update(edge[1].tolist())

    # Pull only OpenPedCan observations touching genes in the drug-target graph.
    gene_targets = set(selected["gene"])
    mutation_relation = ("sample", "has_gene_variant", "gene")
    cnv_relation = ("tumor_type", "copy_number_in_tumor", "gene")
    fusion_gene_relation = ("fusion", "fusion_gene", "gene")
    sample_fusion_relation = ("sample", "has_fusion", "fusion")

    if mutation_relation in raw_edges:
        edge = raw_edges[mutation_relation]
        selected["sample"].update(
            int(sample) for sample, gene in edge.t().tolist()
            if int(gene) in gene_targets
        )
    if cnv_relation in raw_edges:
        edge = raw_edges[cnv_relation]
        selected["tumor_type"].update(
            int(tumor) for tumor, gene in edge.t().tolist()
            if int(gene) in gene_targets
        )
    if fusion_gene_relation in raw_edges:
        edge = raw_edges[fusion_gene_relation]
        selected["fusion"].update(
            int(fusion) for fusion, gene in edge.t().tolist()
            if int(gene) in gene_targets
        )

    # Close the selected sample/fusion neighbourhood in both directions.
    if sample_fusion_relation in raw_edges:
        for _ in range(2):
            for sample, fusion in raw_edges[sample_fusion_relation].t().tolist():
                if int(fusion) in selected["fusion"]:
                    selected["sample"].add(int(sample))
                if int(sample) in selected["sample"]:
                    selected["fusion"].add(int(fusion))

    # Retain histology and sample annotations for selected patient samples.
    sample_tumor_relation = ("sample", "classified_as", "tumor_type")
    if sample_tumor_relation in raw_edges:
        for sample, tumor in raw_edges[sample_tumor_relation].t().tolist():
            if int(sample) in selected["sample"] or int(tumor) in selected["tumor_type"]:
                selected["sample"].add(int(sample))
                selected["tumor_type"].add(int(tumor))

    for relation in (
        ("sample", "has_diagnosis", "diagnosis"),
        ("sample", "from_cohort", "cohort"),
        ("sample", "has_molecular_subtype", "molecular_subtype"),
    ):
        if relation not in raw_edges:
            continue
        for sample, metadata_node in raw_edges[relation].t().tolist():
            if int(sample) in selected["sample"]:
                selected[relation[2]].add(int(metadata_node))

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
    for type_index, node_type in enumerate(node_types):
        offsets[node_type] = running
        count = len(selected_ids[node_type])
        node_counts[node_type] = count
        node_type_values.extend([type_index] * count)
        running += count

    all_edges: list[torch.Tensor] = []
    all_edge_types: list[torch.Tensor] = []
    relation_counts: dict[str, int] = {}
    for relation_index, relation in enumerate(relation_names):
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
        all_edge_types.extend((
            torch.full((len(kept),), relation_index, dtype=torch.long),
            torch.full((len(kept),), relation_index + len(relation_names), dtype=torch.long),
        ))

    if all_edges:
        edge_index_out = torch.cat(all_edges, dim=1).contiguous()
        edge_type_out = torch.cat(all_edge_types, dim=0).contiguous()
    else:
        edge_index_out = torch.empty((2, 0), dtype=torch.long)
        edge_type_out = torch.empty((0,), dtype=torch.long)

    drug_ids = [node_ids["drug"][index] for index in selected_ids["drug"]]
    drug_indices = torch.arange(node_counts["drug"], dtype=torch.long)
    if edge_index_out.numel() and int(edge_index_out.max()) >= running:
        raise ValueError("Compact graph contains an out-of-range node index.")

    return CompactDrugGraph(
        edge_index=edge_index_out,
        edge_type=edge_type_out,
        node_type=torch.tensor(node_type_values, dtype=torch.long),
        drug_indices=drug_indices,
        drug_ids=drug_ids,
        num_nodes=running,
        num_relations=2 * len(relation_names),
        node_counts=node_counts,
        relation_counts=relation_counts,
        node_types=node_types,
        relation_names=relation_names,
    )


EVIDENCE_LEVEL_RELATIONS = {
    "chembl_mapping": set(RELATIONS[:1]),
    "target_evidence": set(RELATIONS[:2]),
    "gene_evidence": set(RELATIONS),
    "pediatric_context": set((*RELATIONS, *OPENPEDCAN_RELATIONS)),
}


def restrict_compact_drug_graph(
    graph: CompactDrugGraph,
    evidence_level: str,
) -> CompactDrugGraph:
    """Mask relation families while keeping identical node and encoder dimensions."""
    if evidence_level not in EVIDENCE_LEVEL_RELATIONS:
        raise ValueError(
            f"Unknown evidence level {evidence_level!r}; expected one of "
            f"{sorted(EVIDENCE_LEVEL_RELATIONS)}."
        )
    allowed_names = EVIDENCE_LEVEL_RELATIONS[evidence_level]
    forward_relation_count = len(graph.relation_names)
    if graph.num_relations != 2 * forward_relation_count:
        raise ValueError("Compact graph relation count does not match its relation schema.")

    allowed_relation_ids = {
        index
        for index, relation in enumerate(graph.relation_names)
        if relation in allowed_names
    }
    allowed_relation_ids |= {
        forward_relation_count + index for index in allowed_relation_ids
    }
    keep_mask = torch.zeros_like(graph.edge_type, dtype=torch.bool)
    for relation_id in allowed_relation_ids:
        keep_mask |= graph.edge_type.eq(relation_id)

    relation_counts = {
        "::".join(relation): (
            int(graph.relation_counts.get("::".join(relation), 0))
            if relation in allowed_names else 0
        )
        for relation in graph.relation_names
    }
    return CompactDrugGraph(
        edge_index=graph.edge_index[:, keep_mask].contiguous(),
        edge_type=graph.edge_type[keep_mask].contiguous(),
        node_type=graph.node_type,
        drug_indices=graph.drug_indices,
        drug_ids=list(graph.drug_ids),
        num_nodes=graph.num_nodes,
        num_relations=graph.num_relations,
        node_counts=dict(graph.node_counts),
        relation_counts=relation_counts,
        node_types=graph.node_types,
        relation_names=graph.relation_names,
    )


def summarize_drug_evidence_coverage(graph: CompactDrugGraph) -> dict[str, Any]:
    """Count drug nodes with complete evidence paths in the response-free graph."""
    edge_arrays = {
        relation_index: graph.edge_index[:, graph.edge_type == relation_index]
        for relation_index in range(len(RELATIONS))
    }

    drug_to_molecules: dict[int, set[int]] = {}
    for drug_index, molecule_index in edge_arrays[0].t().tolist():
        drug_to_molecules.setdefault(int(drug_index), set()).add(int(molecule_index))

    molecule_to_targets: dict[int, set[int]] = {}
    for molecule_index, target_index in edge_arrays[1].t().tolist():
        molecule_to_targets.setdefault(int(molecule_index), set()).add(int(target_index))

    target_to_genes: dict[int, set[int]] = {}
    for target_index, gene_index in edge_arrays[2].t().tolist():
        target_to_genes.setdefault(int(target_index), set()).add(int(gene_index))

    drugs_with_molecule = set(drug_to_molecules)
    drugs_with_target: set[int] = set()
    drugs_with_gene: set[int] = set()
    for drug_index, molecules in drug_to_molecules.items():
        targets = {
            target_index
            for molecule_index in molecules
            for target_index in molecule_to_targets.get(molecule_index, set())
        }
        if targets:
            drugs_with_target.add(drug_index)
        if any(target_to_genes.get(target_index) for target_index in targets):
            drugs_with_gene.add(drug_index)

    total_drugs = len(graph.drug_ids)

    def record(indices: set[int]) -> dict[str, Any]:
        return {
            "count": len(indices),
            "fraction_of_all_graph_drugs": (
                float(len(indices) / total_drugs) if total_drugs else 0.0
            ),
            "drug_ids": [graph.drug_ids[index] for index in sorted(indices)],
        }

    return {
        "total_graph_drugs": total_drugs,
        "drug_with_chembl_molecule_mapping": record(drugs_with_molecule),
        "drug_with_direct_target_path": record(drugs_with_target),
        "drug_with_gene_level_path": record(drugs_with_gene),
        "interpretation": (
            "A complete gene-level path is drug -> ChEMBL molecule -> direct "
            "mechanism target -> gene. Counts are graph-evidence coverage, not "
            "proof that the mechanism is active in a cell line."
        ),
    }


class DrugCellLinePairGNN(nn.Module):
    """Five-layer RGCN + cell-expression encoder + pairwise dual-endpoint head.

    Five layers permit OpenPedCan tumour context to reach drug embeddings along
    sample -> tumour/fusion -> gene -> target -> molecule -> drug paths. Optional
    Morgan/descriptor features add chemical structure information to drug nodes.
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
        num_chemical_features: int = 0,
    ) -> None:
        super().__init__()
        if min(
            num_graph_nodes, num_node_types, num_relations, num_drugs,
            num_expression_features, hidden_dim,
        ) <= 0:
            raise ValueError("All model dimensions must be positive.")
        if num_chemical_features < 0:
            raise ValueError("num_chemical_features cannot be negative.")
        self.entity_embedding = nn.Embedding(num_graph_nodes, hidden_dim)
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim)
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.node_type_embedding.weight)

        num_bases = min(4, num_relations)
        self.drug_convs = nn.ModuleList([
            RGCNConv(
                hidden_dim, hidden_dim, num_relations=num_relations,
                num_bases=num_bases,
            )
            for _ in range(5)
        ])
        self.drug_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(5)])
        self.num_message_passing_layers = 5

        self.chemical_encoder = (
            nn.Sequential(
                nn.Linear(num_chemical_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if num_chemical_features > 0 else None
        )

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
        self.num_chemical_features = num_chemical_features

    def encode(
        self,
        expression_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_type: torch.Tensor,
        chemical_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_index = torch.arange(
            self.entity_embedding.num_embeddings,
            device=self.entity_embedding.weight.device,
        )
        graph_x = self.entity_embedding(node_index) + self.node_type_embedding(node_type)
        if self.chemical_encoder is not None:
            if chemical_features is None:
                raise ValueError("Chemical features are required by this model configuration.")
            if chemical_features.shape != (self.num_drugs, self.num_chemical_features):
                raise ValueError(
                    "Chemical feature matrix must align with compact drug order and configured width."
                )
            graph_x = graph_x.clone()
            graph_x[: self.num_drugs] = (
                graph_x[: self.num_drugs] + self.chemical_encoder(chemical_features)
            )
        elif chemical_features is not None:
            raise ValueError("Chemical features supplied to a model without a chemical encoder.")

        hidden = graph_x
        for layer, (convolution, normalization) in enumerate(
            zip(self.drug_convs, self.drug_norms)
        ):
            update = convolution(hidden, edge_index, edge_type)
            hidden = normalization(F.relu(update + hidden))
            if layer < len(self.drug_convs) - 1:
                hidden = F.dropout(hidden, p=self.dropout, training=self.training)
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
