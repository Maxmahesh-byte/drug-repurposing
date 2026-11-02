import csv
import json
from collections import Counter
from pathlib import Path

import torch

graph_path = Path(
    "data/processed/graph_v2/typed_graph_v2.pt"
)
variants_path = Path(
    "data/processed/graph_v2/intergenic_variants.csv"
)
report_path = Path(
    "data/processed/graph_v2/intergenic_v2_compatibility.json"
)

for path in (graph_path, variants_path):
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")

# Load the existing graph for inspection only.
graph = torch.load(
    graph_path,
    map_location="cpu",
    weights_only=False,
)

if "sample" not in graph.node_types:
    raise ValueError(
        f"No 'sample' node type found. Available: {graph.node_types}"
    )

sample_store = graph["sample"]
if "node_ids" not in sample_store:
    raise ValueError("The V2 sample store has no node_ids field.")

graph_sample_ids = {
    str(value).strip()
    for value in sample_store.node_ids
    if str(value).strip()
}

source_sample_ids = set()
unmatched_sample_counts = Counter()
total_records = 0
empty_sample_ids = 0

with variants_path.open(
    "r", newline="", encoding="utf-8-sig"
) as handle:
    reader = csv.DictReader(handle)

    if "sample_id" not in (reader.fieldnames or []):
        raise ValueError("Extracted CSV has no sample_id column.")

    for row in reader:
        total_records += 1
        sample_id = (row.get("sample_id") or "").strip()

        if not sample_id:
            empty_sample_ids += 1
            continue

        source_sample_ids.add(sample_id)

        if sample_id not in graph_sample_ids:
            unmatched_sample_counts[sample_id] += 1

report = {
    "graph_file": str(graph_path),
    "source_file": str(variants_path),
    "records_scanned": total_records,
    "graph_sample_nodes": len(graph_sample_ids),
    "distinct_source_sample_ids": len(source_sample_ids),
    "source_samples_found_in_graph": (
        len(source_sample_ids) - len(unmatched_sample_counts)
    ),
    "unmatched_sample_id_count": len(unmatched_sample_counts),
    "records_with_unmatched_sample_id": sum(
        unmatched_sample_counts.values()
    ),
    "empty_sample_id_records": empty_sample_ids,
    "unmatched_sample_examples": dict(
        unmatched_sample_counts.most_common(25)
    ),
}

report_path.write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8",
)

print("INTERGENIC ↔ GRAPH V2 COMPATIBILITY")
print("=" * 55)
print(f"Records scanned: {total_records:,}")
print(f"Graph sample nodes: {len(graph_sample_ids):,}")
print(f"Distinct source sample IDs: {len(source_sample_ids):,}")
print(
    "Source sample IDs missing from graph: "
    f"{len(unmatched_sample_counts):,}"
)
print(
    "Records with unmatched sample IDs: "
    f"{sum(unmatched_sample_counts.values()):,}"
)
print(f"Empty sample IDs: {empty_sample_ids:,}")
print(f"\nReport: {report_path}")
print("Graph V2 and source data were not modified.")
