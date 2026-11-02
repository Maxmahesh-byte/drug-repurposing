import torch
from pathlib import Path

path = Path("data/processed/graph_v2/typed_graph_v2.pt")
if not path.exists():
    raise FileNotFoundError(path)

data = torch.load(path, map_location="cpu", weights_only=False)

columns = [
    "variant_count",
    "copy_number_count",
    "fusion_count",
    "ln_ic50",
    "auc",
]

print("EDGE ATTRIBUTE AUDIT")
print("=" * 78)

for edge_type in data.edge_types:
    store = data[edge_type]
    count = store.edge_index.shape[1]
    print(f"\n{edge_type}")
    print("  Edge count:", count)

    if "edge_attr_legacy" in store:
        attrs = store.edge_attr_legacy
        for i, name in enumerate(columns):
            values = attrs[:, i]
            finite = torch.isfinite(values)
            valid = values[finite]
            nonzero = int((valid != 0).sum())

            if valid.numel():
                print(
                    f"  {name}: nonzero={nonzero:,}/{count:,}, "
                    f"min={valid.min().item():.4g}, "
                    f"max={valid.max().item():.4g}, "
                    f"mean={valid.double().mean().item():.4g}, "
                    f"nonfinite={(~finite).sum().item()}"
                )
            else:
                print(f"  {name}: no finite values")

    if "edge_weight_legacy" in store:
        weights = store.edge_weight_legacy
        finite = torch.isfinite(weights)
        valid = weights[finite]
        if valid.numel():
            print(
                f"  legacy weight: min={valid.min().item():.4g}, "
                f"max={valid.max().item():.4g}, "
                f"mean={valid.double().mean().item():.4g}, "
                f"nonfinite={(~finite).sum().item()}"
            )

    if "source_edge_id" in store:
        ids = store.source_edge_id
        unique_count = torch.unique(ids).numel()
        print(
            f"  source_edge_id: unique={unique_count:,}/{ids.numel():,}, "
            f"min={ids.min().item() if ids.numel() else 'n/a'}, "
            f"max={ids.max().item() if ids.numel() else 'n/a'}"
        )

mismatch_type = next(
    (
        et for et in data.edge_types
        if et[1] == "legacy_type_mismatch__has_diagnosis"
    ),
    None,
)

if mismatch_type is not None:
    print("\nMISMATCHED EDGE EXAMPLES")
    print("=" * 78)

    store = data[mismatch_type]
    source_indices, target_indices = store.edge_index.tolist()
    source_ids = list(data[mismatch_type[0]].node_ids)
    target_ids = list(data[mismatch_type[2]].node_ids)
    legacy_ids = store.source_edge_id.tolist()

    for i in range(min(20, len(legacy_ids))):
        print(
            f"legacy_edge={legacy_ids[i]} | "
            f"source={source_ids[source_indices[i]]!r} | "
            f"target={target_ids[target_indices[i]]!r}"
        )

print("\nAudit finished. No graph files were modified.")
