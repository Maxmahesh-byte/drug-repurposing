import csv
import json
from collections import Counter
from pathlib import Path

import torch

base = Path("data/processed/graph_v3_prototype")
graph_path = base / "intergenic_observations_100k.pt"
evidence_path = base / "intergenic_evidence_100k.csv"
manifest_path = base / "intergenic_100k_prototype_manifest.json"

for path in (graph_path, evidence_path, manifest_path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing output: {path}")

graph = torch.load(graph_path, map_location="cpu", weights_only=False)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

sample_ids = list(graph["sample"].node_ids)
observations = graph["intergenic_observation"]
observation_ids = list(observations.node_ids)
edge = graph[
    "sample", "has_intergenic_observation", "intergenic_observation"
].edge_index

with evidence_path.open(encoding="utf-8", newline="") as f:
    evidence = list(csv.DictReader(f))

source_rows = [int(r["source_maf_data_row"]) for r in evidence]
sample_counts = Counter(r["Tumor_Sample_Barcode"] for r in evidence)

checks = {}

checks["sample_nodes_unique"] = (
    len(sample_ids) == len(set(sample_ids)) == 3_223
)
checks["observation_nodes_count"] = len(observation_ids) == 100_000
checks["observation_ids_unique"] = (
    len(observation_ids) == len(set(observation_ids))
)
checks["evidence_rows_count"] = len(evidence) == 100_000
checks["edge_shape"] = tuple(edge.shape) == (2, 100_000)
checks["edge_indices_in_range"] = (
    edge.numel() > 0
    and int(edge.min()) >= 0
    and int(edge[0].max()) < len(sample_ids)
    and int(edge[1].max()) < len(observation_ids)
)
checks["evidence_sample_coverage"] = (
    len(sample_counts) == 3_223
    and set(sample_counts).issubset(set(sample_ids))
)
checks["observation_ids_match_source_rows"] = (
    observation_ids == [
        f"openpedcan_v15_maf_row:{r}" for r in source_rows
    ]
)
checks["source_rows_unique_and_ordered"] = (
    len(source_rows) == len(set(source_rows))
    and source_rows == sorted(source_rows)
)

# Verify that every edge's sample endpoint matches its CSV record.
checks["edge_sample_links_match_evidence"] = all(
    sample_ids[sample_idx]
    == evidence[obs_idx]["Tumor_Sample_Barcode"]
    for sample_idx, obs_idx in edge.T.tolist()
)

checks["correct_source_annotations"] = all(
    r["NCBI_Build"] == "GRCh38"
    and r["Variant_Classification"] == "IGR"
    and r["Hugo_Symbol"] == "Unknown"
    for r in evidence
)

checks["manifest_counts_match"] = (
    manifest.get("records_in_prototype") == 100_000
    and manifest.get("sample_nodes") == len(sample_ids)
    and manifest.get("observation_nodes") == len(observation_ids)
    and manifest.get("observation_edges") == edge.shape[1]
)

print("GRAPH V3 100K VALIDATION")
print("=" * 45)
print(f"Sample nodes: {len(sample_ids):,}")
print(f"Observation nodes: {len(observation_ids):,}")
print(f"Edges: {edge.shape[1]:,}")
print(f"Evidence rows: {len(evidence):,}")
print(f"Samples represented: {len(sample_counts):,}")

for name, passed in checks.items():
    print(f"{name}: {'PASS' if passed else 'FAIL'}")

print(
    "\nRESULT:",
    "PASS" if all(checks.values()) else "INVESTIGATE",
)
