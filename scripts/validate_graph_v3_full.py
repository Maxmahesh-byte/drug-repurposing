import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

V2_PATH = Path("data/processed/graph_v2/typed_graph_v2.pt")
V3_PATH = Path("data/processed/graph_v3/typed_graph_v3.pt")
SOURCE = Path("data/processed/graph_v2/intergenic_variants_maf.csv.gz")
MANIFEST_PATH = Path("data/processed/graph_v3/graph_v3_manifest.json")

EXPECTED_ROWS = 4_522_696
CHUNK_SIZE = 50_000

NEW_NODE = "intergenic_observation"
NEW_EDGE = ("sample", "has_intergenic_observation", NEW_NODE)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_nodes(graph, node_type):
    value = graph[node_type].num_nodes
    if value is not None:
        return int(value)
    return len(graph[node_type].node_ids)


def compare_value(a, b):
    if torch.is_tensor(a) and torch.is_tensor(b):
        return (
            a.dtype == b.dtype
            and tuple(a.shape) == tuple(b.shape)
            and torch.equal(a, b)
        )
    return type(a) is type(b) and a == b


def compare_store(name, old_store, new_store):
    old_keys = set(old_store.keys())
    new_keys = set(new_store.keys())

    if old_keys != new_keys:
        raise AssertionError(
            f"{name}: attribute keys differ: "
            f"old-only={old_keys - new_keys}, "
            f"new-only={new_keys - old_keys}"
        )

    for key in old_keys:
        if not compare_value(old_store[key], new_store[key]):
            raise AssertionError(
                f"{name}: existing attribute {key!r} changed."
            )


def flush_chunk(rows, offset, arrays, edges):
    if not rows:
        return offset

    chunk = np.asarray(rows, dtype=np.int64)
    end = offset + len(chunk)

    # Columns: source row, start, end, sample index,
    # chromosome code, variant-type code.
    names = [
        "source_maf_data_row",
        "start_position",
        "end_position",
    ]

    for column, name in enumerate(names):
        if not np.array_equal(chunk[:, column], arrays[name][offset:end]):
            raise AssertionError(
                f"Source-to-graph mismatch in {name}, "
                f"rows {offset}:{end}"
            )

    if not np.array_equal(chunk[:, 3], edges[0, offset:end]):
        raise AssertionError(
            f"Sample edge mismatch in rows {offset}:{end}"
        )

    if not np.array_equal(
        chunk[:, 4], arrays["chromosome_code"][offset:end]
    ):
        raise AssertionError(
            f"Chromosome code mismatch in rows {offset}:{end}"
        )

    if not np.array_equal(
        chunk[:, 5], arrays["variant_type_code"][offset:end]
    ):
        raise AssertionError(
            f"Variant-type code mismatch in rows {offset}:{end}"
        )

    expected_observation_indices = np.arange(offset, end, dtype=np.int64)
    if not np.array_equal(
        edges[1, offset:end], expected_observation_indices
    ):
        raise AssertionError(
            f"Observation edge indices mismatch in rows {offset}:{end}"
        )

    return end


def main():
    for path in (V2_PATH, V3_PATH, SOURCE, MANIFEST_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    print("Loading Graph V2 and Graph V3...")
    v2 = torch.load(V2_PATH, map_location="cpu", weights_only=False)
    v3 = torch.load(V3_PATH, map_location="cpu", weights_only=False)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    # 1. Verify all original node and edge stores remain present.
    expected_nodes = set(v2.node_types) | {NEW_NODE}
    expected_edges = set(v2.edge_types) | {NEW_EDGE}

    assert set(v3.node_types) == expected_nodes
    assert set(v3.edge_types) == expected_edges

    for node_type in v2.node_types:
        assert count_nodes(v2, node_type) == count_nodes(v3, node_type)
        compare_store(
            f"node store {node_type}",
            v2[node_type],
            v3[node_type],
        )

    for edge_type in v2.edge_types:
        assert (
            v2[edge_type].edge_index.shape
            == v3[edge_type].edge_index.shape
        )
        compare_store(
            f"edge store {edge_type}",
            v2[edge_type],
            v3[edge_type],
        )

    print("Original Graph V2 node stores: PASS")
    print("Original Graph V2 edge stores: PASS")

    # 2. Check the new node and relation structure.
    obs = v3[NEW_NODE]
    edge_index = v3[NEW_EDGE].edge_index

    assert count_nodes(v3, NEW_NODE) == EXPECTED_ROWS
    assert tuple(edge_index.shape) == (2, EXPECTED_ROWS)

    required_attributes = [
        "source_maf_data_row",
        "start_position",
        "end_position",
        "chromosome_code",
        "variant_type_code",
    ]
    for name in required_attributes:
        assert name in obs, f"Missing new node attribute: {name}"
        assert obs[name].numel() == EXPECTED_ROWS

    arrays = {
        name: obs[name].cpu().numpy()
        for name in required_attributes
    }
    edges = edge_index.cpu().numpy()

    assert np.all(np.diff(arrays["source_maf_data_row"]) > 0)
    assert np.all(arrays["start_position"] > 0)
    assert np.all(
        arrays["end_position"] >= arrays["start_position"]
    )
    assert np.all(edges[0] >= 0)
    assert np.all(edges[0] < count_nodes(v3, "sample"))
    assert np.array_equal(
        edges[1], np.arange(EXPECTED_ROWS, dtype=np.int64)
    )

    sample_ids = [str(x) for x in v3["sample"].node_ids]
    sample_to_index = {
        sample_id: i for i, sample_id in enumerate(sample_ids)
    }
    assert len(sample_to_index) == len(sample_ids)

    chromosome_codes = manifest["chromosome_code_map"]
    variant_type_codes = manifest["variant_type_code_map"]

    for code in arrays["chromosome_code"]:
        assert int(code) in set(chromosome_codes.values())
    for code in arrays["variant_type_code"]:
        assert int(code) in set(variant_type_codes.values())

    print("New graph node attributes and edge ranges: PASS")

    # 3. Compare each observation to the source extract in batches.
    source_hash = sha256_file(SOURCE)
    assert source_hash == manifest["source_evidence_sha256"]

    rows = []
    offset = 0
    represented_samples = set()

    with gzip.open(
        SOURCE, "rt", encoding="utf-8", newline=""
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            if (
                row["NCBI_Build"] != "GRCh38"
                or row["Variant_Classification"] != "IGR"
                or row["Hugo_Symbol"] != "Unknown"
            ):
                raise AssertionError(
                    "Unexpected source annotation at "
                    f"source row {row['source_maf_data_row']}"
                )

            sample_id = row["Tumor_Sample_Barcode"]
            if sample_id not in sample_to_index:
                raise AssertionError(
                    f"Missing graph sample node: {sample_id}"
                )
            represented_samples.add(sample_id)

            chromosome = row["Chromosome"]
            variant_type = row["Variant_Type"]

            if chromosome not in chromosome_codes:
                raise AssertionError(f"Unknown chromosome: {chromosome}")
            if variant_type not in variant_type_codes:
                raise AssertionError(f"Unknown variant type: {variant_type}")

            rows.append([
                int(row["source_maf_data_row"]),
                int(row["Start_Position"]),
                int(row["End_Position"]),
                sample_to_index[sample_id],
                chromosome_codes[chromosome],
                variant_type_codes[variant_type],
            ])

            if len(rows) == CHUNK_SIZE:
                offset = flush_chunk(rows, offset, arrays, edges)
                rows.clear()

        offset = flush_chunk(rows, offset, arrays, edges)

    assert offset == EXPECTED_ROWS
    assert len(represented_samples) == 3_223

    # The evidence file must not have changed during validation.
    assert sha256_file(SOURCE) == source_hash

    print("Full source-to-graph alignment: PASS")
    print(f"Source records verified: {offset:,}")
    print(f"Distinct sample barcodes: {len(represented_samples):,}")

    # 4. Confirm manifest summary values.
    assert manifest["source_rows"] == EXPECTED_ROWS
    assert manifest["new_node_count"] == EXPECTED_ROWS
    assert manifest["new_edge_count"] == EXPECTED_ROWS
    assert manifest["sample_nodes_matched"] == 3_223

    print("Manifest consistency: PASS")
    print("\nRESULT: PASS")


if __name__ == "__main__":
    main()
