import csv
import torch
from pathlib import Path

graph_path = Path("data/processed/graph_v2/typed_graph_v2.pt")
data = torch.load(
    graph_path,
    map_location="cpu",
    weights_only=False,
)

edge_type = ("sample", "has_gene_variant", "gene")
store = data[edge_type]

counts = store.edge_attr_legacy[:, 0]
src, dst = store.edge_index
sample_ids = list(data["sample"].node_ids)
gene_ids = list(data["gene"].node_ids)
legacy_ids = store.source_edge_id

top_n = min(25, counts.numel())
values, indices = torch.topk(counts, top_n)

print("TOP SAMPLE–GENE MUTATION COUNTS")
print("=" * 80)

for rank, (value, idx) in enumerate(zip(values.tolist(), indices.tolist()), 1):
    print(
        f"{rank:02d}. count={value:g} | "
        f"sample={sample_ids[src[idx].item()]!r} | "
        f"gene={gene_ids[dst[idx].item()]!r} | "
        f"legacy_edge={legacy_ids[idx].item()}"
    )

print("\nSOURCE FILE CHECK")
print("=" * 80)

candidates = [
    Path("data/raw/openpedcan/openpedcan_mutations_phase2.csv"),
    Path("data/raw/openpedcan/openpedcan_phase1.csv"),
    Path("data/raw/gdsc/gdsc2_responses.csv"),
]

for path in candidates:
    print(f"\n{path}: {'FOUND' if path.exists() else 'NOT FOUND'}")
    if path.exists() and path.suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            print("  Header:", next(reader, []))

print("\nInspection complete. No files were modified.")
