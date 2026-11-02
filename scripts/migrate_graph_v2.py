#!/usr/bin/env python3
"""Create a typed PyTorch Geometric view of the existing research graph.

This is a non-destructive migration step. It reads data/processed/openpedcan_graph.pt
and writes data/processed/graph_v2/typed_graph_v2.pt plus a manifest. It does not
re-download data or overwrite the existing graph.

Important: this makes node/edge types explicit, but cannot recover source metadata
that the original graph builder did not save. Raw CSVs remain the detailed evidence.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "openpedcan_graph.pt"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "graph_v2"

# Canonical orientation for each relation, independent of original undirected
# NetworkX endpoint order. The script validates these assumptions against data.
RELATION_SCHEMA: dict[str, tuple[str, str]] = {
    "chembl_direct_mechanism": ("chembl_molecule", "target"),
    "classified_as": ("sample", "tumor_type"),
    "copy_number_in_tumor": ("tumor_type", "gene"),
    "exact_name_chembl_mapping": ("drug", "chembl_molecule"),
    "from_cohort": ("sample", "cohort"),
    "fusion_gene": ("fusion", "gene"),
    "gdsc2_response": ("drug", "cell_line"),
    "gdsc_putative_pathway_annotation": ("drug", "putative_pathway_annotation"),
    "gdsc_putative_target_annotation": ("drug", "putative_target_annotation"),
    "has_chembl_gene_symbol": ("target", "gene"),
    "has_diagnosis": ("sample", "diagnosis"),
    "has_fusion": ("sample", "fusion"),
    "has_gene_variant": ("sample", "gene"),
    "has_molecular_subtype": ("sample", "molecular_subtype"),
    "has_reported_cancer_type": ("cell_line", "reported_cancer_type"),
}

# Relation-level provenance is explicit. Aggregation notes document what was
# encoded by the original builder; consult the raw source table for full detail.
RELATION_PROVENANCE: dict[str, dict[str, str]] = {
    "chembl_direct_mechanism": {
        "source_file": "data/raw/chembl/gdsc2_chembl_target_evidence.csv",
        "derivation": "ChEMBL molecule-to-target mechanism relation; original graph may collapse repeated molecule-target pairs.",
    },
    "classified_as": {
        "source_file": "data/raw/openpedcan/openpedcan_phase1.csv",
        "derivation": "Sample-to-tumour-type classification from OpenPedCan sample metadata.",
    },
    "copy_number_in_tumor": {
        "source_file": "data/raw/openpedcan/openpedcan_copy_number_phase3.csv",
        "derivation": "Counts aggregated by tumour type and gene; not sample-specific copy-number measurements.",
    },
    "exact_name_chembl_mapping": {
        "source_file": "data/raw/chembl/gdsc2_chembl_target_evidence.csv",
        "derivation": "GDSC drug ID to ChEMBL molecule mapping represented in the original graph; consult drug_match_method in the source table.",
    },
    "from_cohort": {
        "source_file": "data/raw/openpedcan/openpedcan_phase1.csv",
        "derivation": "Sample-to-cohort membership.",
    },
    "fusion_gene": {
        "source_file": "data/raw/openpedcan/openpedcan_fusions_phase4.csv",
        "derivation": "Fusion entity to partner-gene associations.",
    },
    "gdsc2_response": {
        "source_file": "data/raw/gdsc/gdsc2_responses.csv",
        "derivation": "Response values aggregated by model, cell line, reported cancer type and drug; ln_ic50 and AUC are means in the legacy edge attributes.",
    },
    "gdsc_putative_pathway_annotation": {
        "source_file": "data/raw/gdsc/gdsc2_responses.csv",
        "derivation": "Drug-to-putative-pathway annotations from GDSC table fields; annotation should not be treated as independently validated pathway evidence.",
    },
    "gdsc_putative_target_annotation": {
        "source_file": "data/raw/gdsc/gdsc2_responses.csv",
        "derivation": "Drug-to-putative-target annotations from GDSC table fields; annotation should not be treated as independently validated target evidence.",
    },
    "has_chembl_gene_symbol": {
        "source_file": "data/raw/chembl/gdsc2_chembl_target_evidence.csv",
        "derivation": "ChEMBL target-to-gene-symbol association.",
    },
    "has_diagnosis": {
        "source_file": "data/raw/openpedcan/openpedcan_phase1.csv",
        "derivation": "Sample-to-diagnosis association.",
    },
    "has_fusion": {
        "source_file": "data/raw/openpedcan/openpedcan_fusions_phase4.csv",
        "derivation": "Sample-to-fusion association. Fusion identifier includes sample, fusion name and breakpoint location.",
    },
    "has_gene_variant": {
        "source_file": "data/raw/openpedcan/openpedcan_mutations_phase2.csv",
        "derivation": "Mutation rows counted by sample and gene; individual variant details remain in the raw source table.",
    },
    "has_molecular_subtype": {
        "source_file": "data/raw/openpedcan/openpedcan_phase1.csv",
        "derivation": "Sample-to-molecular-subtype association.",
    },
    "has_reported_cancer_type": {
        "source_file": "data/raw/gdsc/gdsc2_responses.csv",
        "derivation": "GDSC model-to-reported-cancer-type annotation.",
    },
}

LEGACY_EDGE_ATTR_COLUMNS = [
    "variant_count",
    "copy_number_count",
    "fusion_count",
    "ln_ic50",
    "auc",
]


def _load_graph(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Input graph not found: {path}")
    # Only load a graph you trust as a locally generated artifact. torch.load uses
    # Python deserialization; do not run this on an untrusted .pt file.
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> int:
    input_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_INPUT
    output_dir = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = _load_graph(input_path)
    required = ("x", "edge_index", "edge_type", "edge_attr", "edge_weight", "node_ids", "node_type_names", "edge_relation_names")
    missing = [name for name in required if not hasattr(graph, name)]
    if missing:
        raise ValueError(f"Input graph is missing required attributes: {missing}")

    node_ids = list(graph.node_ids)
    node_type_names = list(graph.node_type_names)
    edge_relation_names = list(graph.edge_relation_names)
    n_nodes = int(graph.num_nodes)
    n_edges = int(graph.edge_index.shape[1])

    if len(node_ids) != n_nodes:
        raise ValueError(f"node_ids length {len(node_ids)} does not match num_nodes {n_nodes}")
    if graph.x.ndim != 2 or graph.x.shape[0] != n_nodes:
        raise ValueError(f"Unexpected node feature shape: {tuple(graph.x.shape)}")
    if graph.edge_index.shape != (2, n_edges):
        raise ValueError(f"Unexpected edge_index shape: {tuple(graph.edge_index.shape)}")
    if graph.edge_type.numel() != n_edges:
        raise ValueError("edge_type length does not match edge count")
    if graph.edge_attr.shape[0] != n_edges or graph.edge_weight.numel() != n_edges:
        raise ValueError("edge attributes/weights do not match edge count")
    edge_attribute_columns = list(
        getattr(graph, "edge_attr_names", LEGACY_EDGE_ATTR_COLUMNS)
    )
    if len(edge_attribute_columns) != graph.edge_attr.shape[1]:
        raise ValueError(
            "Edge attribute column names do not match the edge_attr width: "
            f"{len(edge_attribute_columns)} names for {graph.edge_attr.shape[1]} values."
        )
    if not torch.isfinite(graph.x).all().item():
        raise ValueError("Node feature tensor contains non-finite values")
    if not torch.isfinite(graph.edge_attr).all().item():
        raise ValueError("Edge attribute tensor contains non-finite values")
    if not torch.isfinite(graph.edge_weight).all().item():
        raise ValueError("Edge weight tensor contains non-finite values")

    # The legacy feature matrix was confirmed to be one-hot node-type encoding.
    # Verify instead of silently assuming the first columns match node_type_names.
    type_width = len(node_type_names)
    if graph.x.shape[1] < type_width:
        raise ValueError("Node feature width is smaller than the number of node types")
    x_type = graph.x[:, :type_width]
    if not torch.all((x_type == 0) | (x_type == 1)).item():
        raise ValueError("Expected binary one-hot node-type columns in the legacy x tensor")
    if not torch.all(x_type.sum(dim=1) == 1).item():
        raise ValueError("Each node must have exactly one active type feature")
    if graph.x.shape[1] > type_width and torch.any(graph.x[:, type_width:] != 0).item():
        raise ValueError("Unexpected non-zero legacy node features beyond node-type columns")
    node_type_index = x_type.argmax(dim=1).to(torch.long)

    if len(set(node_ids)) != len(node_ids):
        raise ValueError("Legacy graph contains duplicate global node IDs")

    unknown_relations = set(edge_relation_names) - set(RELATION_SCHEMA)
    if unknown_relations:
        raise ValueError(f"No orientation schema defined for relations: {sorted(unknown_relations)}")
    missing_relations = set(RELATION_SCHEMA) - set(edge_relation_names)
    if missing_relations:
        raise ValueError(f"Expected relations missing from input graph: {sorted(missing_relations)}")
    schema_types = {
        endpoint_type
        for pair in RELATION_SCHEMA.values()
        for endpoint_type in pair
    }
    missing_node_types = schema_types - set(node_type_names)
    if missing_node_types:
        raise ValueError(f"Relation schema refers to missing node types: {sorted(missing_node_types)}")

    # Create a local node-index space for each node type. Original identifiers are
    # retained verbatim, but distinct node stores mean types are naturally namespaced.
    hetero = HeteroData()
    global_to_local = torch.empty(n_nodes, dtype=torch.long)
    node_counts: dict[str, int] = {}
    type_to_id = {name: i for i, name in enumerate(node_type_names)}

    for node_type in node_type_names:
        type_id = type_to_id[node_type]
        global_indices = torch.nonzero(node_type_index == type_id, as_tuple=False).view(-1)
        global_to_local[global_indices] = torch.arange(global_indices.numel(), dtype=torch.long)
        store = hetero[node_type]
        store.num_nodes = int(global_indices.numel())
        store.node_ids = [node_ids[i] for i in global_indices.tolist()]
        node_counts[node_type] = int(global_indices.numel())

    relation_counts: dict[str, int] = {}
    typed_edge_store_counts: dict[str, int] = {}
    relation_pair_checks: dict[str, dict] = {}
    mismatch_edge_total = 0
    relation_id_by_name = {name: idx for idx, name in enumerate(edge_relation_names)}
    mismatch_relation_names: set[str] = set()

    for relation in edge_relation_names:
        relation_id = relation_id_by_name[relation]
        positions = torch.nonzero(graph.edge_type == relation_id, as_tuple=False).view(-1)
        expected_source, expected_target = RELATION_SCHEMA[relation]
        expected_source_id = type_to_id[expected_source]
        expected_target_id = type_to_id[expected_target]
        left = graph.edge_index[0, positions]
        right = graph.edge_index[1, positions]
        left_types = node_type_index[left]
        right_types = node_type_index[right]
        forward = (left_types == expected_source_id) & (right_types == expected_target_id)
        reverse = (left_types == expected_target_id) & (right_types == expected_source_id)
        valid = forward | reverse

        # Edges matching the declared relation schema are stored under the expected
        # endpoint types and oriented canonically, regardless of NetworkX edge order.
        if valid.any().item():
            valid_positions = positions[valid]
            src_global = torch.where(forward[valid], left[valid], right[valid])
            dst_global = torch.where(forward[valid], right[valid], left[valid])
            edge_store = hetero[(expected_source, relation, expected_target)]
            edge_store.edge_index = torch.stack(
                (global_to_local[src_global], global_to_local[dst_global]), dim=0
            ).contiguous()
            edge_store.edge_attr_legacy = graph.edge_attr[valid_positions].contiguous()
            edge_store.edge_weight_legacy = graph.edge_weight[valid_positions].contiguous()
            edge_store.source_edge_id = valid_positions.contiguous()
            key = f"{expected_source} -[{relation}]-> {expected_target}"
            typed_edge_store_counts[key] = int(valid_positions.numel())

        # The old simple graph has already merged some same-string node labels and
        # may have overwritten endpoint type/edge metadata. Do not guess which type
        # was intended. Preserve each contradictory edge in a clearly flagged edge
        # store whose endpoint types match the node types actually saved in the file.
        mismatch_positions = positions[~valid]
        mismatch_details: list[dict[str, object]] = []
        if mismatch_positions.numel():
            mismatch_edge_total += int(mismatch_positions.numel())
            mismatch_left_types = left_types[~valid]
            mismatch_right_types = right_types[~valid]
            mismatch_left_global = left[~valid]
            mismatch_right_global = right[~valid]
            encoded_pairs = (
                mismatch_left_types * len(node_type_names) + mismatch_right_types
            )
            unique_pairs, pair_counts = torch.unique(encoded_pairs, return_counts=True)
            # Several input endpoint orders can canonicalize to the same typed edge
            # store. Accumulate chunks so none are overwritten by a later chunk.
            mismatch_buckets: dict[
                tuple[str, str, str],
                list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
            ] = {}
            for encoded, pair_count in zip(unique_pairs.tolist(), pair_counts.tolist()):
                actual_left_id = encoded // len(node_type_names)
                actual_right_id = encoded % len(node_type_names)
                actual_left_type = node_type_names[actual_left_id]
                actual_right_type = node_type_names[actual_right_id]
                pair_mask = (
                    (mismatch_left_types == actual_left_id)
                    & (mismatch_right_types == actual_right_id)
                )
                pair_positions = mismatch_positions[pair_mask]
                pair_left = mismatch_left_global[pair_mask]
                pair_right = mismatch_right_global[pair_mask]

                # Prefer the schema's expected source endpoint if it is present;
                # otherwise prefer the expected target endpoint. If neither is
                # present, use stable alphabetical ordering for reproducibility.
                if actual_left_type == expected_source:
                    src_global, dst_global = pair_left, pair_right
                    actual_source_type, actual_target_type = actual_left_type, actual_right_type
                elif actual_right_type == expected_source:
                    src_global, dst_global = pair_right, pair_left
                    actual_source_type, actual_target_type = actual_right_type, actual_left_type
                elif actual_right_type == expected_target:
                    src_global, dst_global = pair_left, pair_right
                    actual_source_type, actual_target_type = actual_left_type, actual_right_type
                elif actual_left_type == expected_target:
                    src_global, dst_global = pair_right, pair_left
                    actual_source_type, actual_target_type = actual_right_type, actual_left_type
                elif actual_left_type <= actual_right_type:
                    src_global, dst_global = pair_left, pair_right
                    actual_source_type, actual_target_type = actual_left_type, actual_right_type
                else:
                    src_global, dst_global = pair_right, pair_left
                    actual_source_type, actual_target_type = actual_right_type, actual_left_type

                flagged_relation = f"legacy_type_mismatch__{relation}"
                mismatch_relation_names.add(flagged_relation)
                store_key = (actual_source_type, flagged_relation, actual_target_type)
                mismatch_buckets.setdefault(store_key, []).append(
                    (global_to_local[src_global], global_to_local[dst_global], pair_positions)
                )
                mismatch_details.append({
                    "actual_endpoint_types_in_legacy_order": [actual_left_type, actual_right_type],
                    "migrated_endpoint_types": [actual_source_type, actual_target_type],
                    "count": int(pair_count),
                    "migration_relation_name": flagged_relation,
                })

            for (actual_source_type, flagged_relation, actual_target_type), chunks in mismatch_buckets.items():
                all_src = torch.cat([chunk[0] for chunk in chunks])
                all_dst = torch.cat([chunk[1] for chunk in chunks])
                all_positions = torch.cat([chunk[2] for chunk in chunks])
                edge_store = hetero[(actual_source_type, flagged_relation, actual_target_type)]
                edge_store.edge_index = torch.stack((all_src, all_dst), dim=0).contiguous()
                edge_store.edge_attr_legacy = graph.edge_attr[all_positions].contiguous()
                edge_store.edge_weight_legacy = graph.edge_weight[all_positions].contiguous()
                edge_store.source_edge_id = all_positions.contiguous()
                key = f"{actual_source_type} -[{flagged_relation}]-> {actual_target_type}"
                typed_edge_store_counts[key] = int(all_positions.numel())

        relation_counts[relation] = int(positions.numel())
        relation_pair_checks[relation] = {
            "expected_endpoint_types": [expected_source, expected_target],
            "forward_order_in_input": int(forward.sum().item()),
            "reverse_order_in_input": int(reverse.sum().item()),
            "type_mismatch_edges_preserved": int(mismatch_positions.numel()),
            "type_mismatch_endpoint_type_pairs": mismatch_details,
        }

    # Confirm every edge from the legacy graph appears exactly once in the typed view.
    migrated_edge_total = sum(relation_counts.values())
    if migrated_edge_total != n_edges:
        raise AssertionError(f"Migrated {migrated_edge_total} edges, expected {n_edges}")

    hetero.schema_version = "2.0.0-migration"
    hetero.source_graph = str(input_path.relative_to(ROOT)) if input_path.is_relative_to(ROOT) else str(input_path)
    hetero.node_type_names = node_type_names
    hetero.edge_relation_names = edge_relation_names
    hetero.mismatch_edge_relation_names = sorted(mismatch_relation_names)
    hetero.type_mismatch_edge_count = int(mismatch_edge_total)
    hetero.legacy_node_type_features_removed = True
    hetero.legacy_edge_attributes_preserved_as = "edge_attr_legacy"
    hetero.legacy_edge_attribute_columns = edge_attribute_columns
    hetero.legacy_edge_weights_preserved_as = "edge_weight_legacy"

    output_graph = output_dir / "typed_graph_v2.pt"
    manifest_path = output_dir / "graph_v2_manifest.json"
    torch.save(hetero, output_graph)

    manifest = {
        "schema_version": "2.0.0-migration",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_graph": str(input_path.relative_to(ROOT)) if input_path.is_relative_to(ROOT) else str(input_path),
        "output_graph": str(output_graph.relative_to(ROOT)) if output_graph.is_relative_to(ROOT) else str(output_graph),
        "migration_scope": "Typed view of the existing graph; no raw datasets downloaded or overwritten.",
        "node_count": n_nodes,
        "edge_count": n_edges,
        "node_counts_by_type": node_counts,
        "edge_counts_by_relation": relation_counts,
        "edge_counts_by_typed_edge_store": typed_edge_store_counts,
        "relation_endpoint_orientation_checks": relation_pair_checks,
        "type_mismatch_edge_count": int(mismatch_edge_total),
        "type_mismatch_handling": "Edges contradicting the expected relation endpoint schema are preserved under legacy_type_mismatch__<relation> stores using the node types actually encoded in the legacy graph. Their intended endpoint types cannot safely be inferred; see per-relation details in relation_endpoint_orientation_checks.",
        "legacy_node_features": {
            "shape": list(graph.x.shape),
            "interpretation": "One-hot node-type encoding only; not biological features. Removed from node stores to avoid implying otherwise.",
        },
        "legacy_edge_attribute_columns": edge_attribute_columns,
        "legacy_edge_attribute_note": "Retained as edge_attr_legacy. Columns mix counts and GDSC response values and are not a universal feature set or confidence score.",
        "legacy_edge_weight_note": "Retained as edge_weight_legacy for traceability only. The original builder combines mutation/copy-number/fusion counts and defaults to 1; do not interpret as calibrated confidence.",
        "known_limitations": [
            "The original graph was constructed with NetworkX Graph, which cannot store multiple independent edges between the same pair of nodes.",
            "Some evidence rows were aggregated in the original builder; this migration cannot recover information already lost during aggregation.",
            "Per-edge ChEMBL metadata such as mechanism/action/source URL was not saved in the original .pt graph. The raw source CSV remains authoritative.",
            "Mutation details remain in the raw mutation CSV; the graph represents sample-gene mutation counts.",
            "Copy-number graph edges are aggregated by tumour type and gene rather than retaining every sample-specific call.",
            "The graph has topology and identifiers but does not contain validated biological node feature vectors.",
            "Some legacy edges have endpoint node types inconsistent with their named relation. These edges are retained under explicitly flagged legacy_type_mismatch__ relation stores rather than silently discarded or guessed; original NetworkX node/edge attribute collisions may be the cause.",
        ],
        "relation_provenance": RELATION_PROVENANCE,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Typed graph migration completed.")
    print(f"Input graph: {input_path}")
    print(f"Output graph: {output_graph}")
    print(f"Manifest: {manifest_path}")
    print(f"Nodes: {n_nodes:,}; edges: {n_edges:,}")
    print("\nNode counts:")
    for name, count in node_counts.items():
        print(f"  {name}: {count:,}")
    print("\nEdge counts:")
    for name, count in relation_counts.items():
        print(f"  {name}: {count:,}")
    print(f"\nEdges with endpoint-type/schema mismatches preserved explicitly: {mismatch_edge_total:,}")
    print("Mismatch edges are stored under legacy_type_mismatch__* relation names; inspect the manifest before modelling.")
    print("Reminder: this creates a typed baseline view, not the final evidence-preserving rebuild.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
