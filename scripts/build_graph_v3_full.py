import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

GRAPH_V2 = Path("data/processed/graph_v2/typed_graph_v2.pt")
SOURCE = Path(
    "data/processed/graph_v2/intergenic_variants_maf.csv.gz"
)
OUT_DIR = Path("data/processed/graph_v3")
GRAPH_OUT = OUT_DIR / "typed_graph_v3.pt"
MANIFEST_OUT = OUT_DIR / "graph_v3_manifest.json"

TEMP_GRAPH = OUT_DIR / ".typed_graph_v3.pt.tmp"
TEMP_MANIFEST = OUT_DIR / ".graph_v3_manifest.json.tmp"

EXPECTED_ROWS = 4_522_696
EXPECTED_SAMPLES = 3_223

NEW_NODE_TYPE = "intergenic_observation"
NEW_EDGE_TYPE = (
    "sample",
    "has_intergenic_observation",
    NEW_NODE_TYPE,
)

REQUIRED_COLUMNS = {
    "source_maf_data_row",
    "Hugo_Symbol",
    "NCBI_Build",
    "Chromosome",
    "Start_Position",
    "End_Position",
    "Variant_Classification",
    "Variant_Type",
    "Reference_Allele",
    "Tumor_Seq_Allele1",
    "Tumor_Seq_Allele2",
    "Tumor_Sample_Barcode",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def edge_count(graph, edge_type):
    return int(graph[edge_type].edge_index.shape[1])


def node_count(graph, node_type):
    value = graph[node_type].num_nodes
    if value is not None:
        return int(value)
    return len(graph[node_type].node_ids)


def validate_source_row(
    row, previous_source_row, sample_to_index,
    chromosome_codes=None, variant_type_codes=None,
):
    required_values = [
        "source_maf_data_row",
        "Hugo_Symbol",
        "NCBI_Build",
        "Chromosome",
        "Start_Position",
        "End_Position",
        "Variant_Classification",
        "Variant_Type",
        "Reference_Allele",
        "Tumor_Seq_Allele1",
        "Tumor_Seq_Allele2",
        "Tumor_Sample_Barcode",
    ]

    missing = [
        field for field in required_values
        if not (row.get(field) or "").strip()
    ]
    if missing:
        raise ValueError(f"Missing fields {missing} in source row")

    source_row = int(row["source_maf_data_row"])
    if source_row <= previous_source_row:
        raise ValueError(
            f"Source row identifiers are not strictly increasing: "
            f"{source_row}"
        )

    if row["NCBI_Build"] != "GRCh38":
        raise ValueError(f"Unexpected genome build at row {source_row}")
    if row["Variant_Classification"] != "IGR":
        raise ValueError(f"Unexpected classification at row {source_row}")
    if row["Hugo_Symbol"] != "Unknown":
        raise ValueError(f"Unexpected gene label at row {source_row}")

    start = int(row["Start_Position"])
    end = int(row["End_Position"])
    if start <= 0 or end < start:
        raise ValueError(f"Invalid coordinates at source row {source_row}")

    sample_id = row["Tumor_Sample_Barcode"]
    if sample_id not in sample_to_index:
        raise ValueError(
            f"Sample {sample_id!r} at source row {source_row} "
            "does not exist in Graph V2"
        )

    if chromosome_codes is not None:
        if row["Chromosome"] not in chromosome_codes:
            raise ValueError(
                f"Unexpected chromosome category: {row['Chromosome']!r}"
            )

    if variant_type_codes is not None:
        if row["Variant_Type"] not in variant_type_codes:
            raise ValueError(
                f"Unexpected variant type: {row['Variant_Type']!r}"
            )

    return source_row, start, end, sample_id


if __name__ == "__main__":
    for path in (GRAPH_V2, SOURCE):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")

    # Never overwrite a previous full build.
    if GRAPH_OUT.exists() or MANIFEST_OUT.exists():
        raise FileExistsError(
            f"Output already exists in {OUT_DIR}; "
            "inspect it before choosing a new output version."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_GRAPH.unlink(missing_ok=True)
    TEMP_MANIFEST.unlink(missing_ok=True)

    print("Loading Graph V2; the original file will not be modified.")
    graph = torch.load(
        GRAPH_V2, map_location="cpu", weights_only=False
    )

    if "sample" not in graph.node_types:
        raise ValueError("Graph V2 is missing its sample node type.")
    if NEW_NODE_TYPE in graph.node_types:
        raise ValueError(
            f"Graph V2 already contains {NEW_NODE_TYPE!r}."
        )
    if NEW_EDGE_TYPE in graph.edge_types:
        raise ValueError(
            f"Graph V2 already contains relation {NEW_EDGE_TYPE}."
        )
    if not hasattr(graph["sample"], "node_ids"):
        raise ValueError("Graph V2 sample nodes have no node_ids field.")

    sample_ids = [str(value) for value in graph["sample"].node_ids]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Graph V2 sample IDs are not unique.")

    sample_to_index = {
        sample_id: i for i, sample_id in enumerate(sample_ids)
    }

    original_node_counts = {
        node_type: node_count(graph, node_type)
        for node_type in graph.node_types
    }
    original_edge_counts = {
        edge_type: edge_count(graph, edge_type)
        for edge_type in graph.edge_types
    }

    source_hash_before = sha256_file(SOURCE)

    # PASS 1: validate every row and discover categorical encodings.
    print("Pass 1/2: validating the complete intergenic extract.")
    row_count = 0
    previous_source_row = 0
    sample_seen = set()
    chromosome_values = set()
    variant_type_values = set()

    with SOURCE.open("rb") as _:
        pass

    import gzip

    with gzip.open(
        SOURCE, "rt", encoding="utf-8", newline=""
    ) as f:
        reader = csv.DictReader(f)
        absent = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
        if absent:
            raise ValueError(f"Missing source columns: {absent}")

        for row in reader:
            source_row, start, end, sample_id = validate_source_row(
                row, previous_source_row, sample_to_index
            )
            previous_source_row = source_row
            row_count += 1
            sample_seen.add(sample_id)
            chromosome_values.add(row["Chromosome"])
            variant_type_values.add(row["Variant_Type"])

    if row_count != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} source records; "
            f"found {row_count:,}."
        )
    if len(sample_seen) != EXPECTED_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLES:,} distinct source samples; "
            f"found {len(sample_seen):,}."
        )

    chromosome_codes = {
        value: index
        for index, value in enumerate(sorted(chromosome_values), start=1)
    }
    variant_type_codes = {
        value: index
        for index, value in enumerate(sorted(variant_type_values), start=1)
    }

    print(f"Validated rows: {row_count:,}")
    print(f"Samples matched to Graph V2: {len(sample_seen):,}")
    print(f"Chromosome categories: {len(chromosome_codes):,}")
    print(f"Variant-type categories: {len(variant_type_codes):,}")

    # Allocate compact numeric arrays. Do not hold CSV rows in memory.
    source_rows = np.empty(row_count, dtype=np.int64)
    starts = np.empty(row_count, dtype=np.int64)
    ends = np.empty(row_count, dtype=np.int64)
    sample_indices = np.empty(row_count, dtype=np.int64)
    chromosome_array = np.empty(row_count, dtype=np.int16)
    variant_type_array = np.empty(row_count, dtype=np.int16)

    # PASS 2: stream source data into preallocated arrays.
    print("Pass 2/2: constructing graph attributes and sample links.")
    written = 0
    previous_source_row = 0

    with gzip.open(
        SOURCE, "rt", encoding="utf-8", newline=""
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            source_row, start, end, sample_id = validate_source_row(
                row,
                previous_source_row,
                sample_to_index,
                chromosome_codes,
                variant_type_codes,
            )
            previous_source_row = source_row

            if written >= row_count:
                raise ValueError("Source record count changed between passes.")

            source_rows[written] = source_row
            starts[written] = start
            ends[written] = end
            sample_indices[written] = sample_to_index[sample_id]
            chromosome_array[written] = chromosome_codes[row["Chromosome"]]
            variant_type_array[written] = variant_type_codes[row["Variant_Type"]]
            written += 1

    if written != row_count:
        raise ValueError(
            f"Second pass produced {written:,} rows, "
            f"expected {row_count:,}."
        )

    source_hash_after = sha256_file(SOURCE)
    if source_hash_before != source_hash_after:
        raise ValueError(
            "The source extract changed during the build; "
            "refusing to publish this graph."
        )

    # Add one node per observed intergenic record.
    # The numeric source-row field is the stable node identifier.
    # Full row strings and alleles remain in the companion source extract.
    obs = graph[NEW_NODE_TYPE]
    obs.num_nodes = row_count
    obs.source_maf_data_row = torch.from_numpy(source_rows)
    obs.start_position = torch.from_numpy(starts)
    obs.end_position = torch.from_numpy(ends)
    obs.chromosome_code = torch.from_numpy(chromosome_array)
    obs.variant_type_code = torch.from_numpy(variant_type_array)

    # One directed edge from the existing sample node to each observation.
    edge_array = np.empty((2, row_count), dtype=np.int64)
    edge_array[0] = sample_indices
    edge_array[1] = np.arange(row_count, dtype=np.int64)
    graph[NEW_EDGE_TYPE].edge_index = torch.from_numpy(edge_array)

    # Verify that every old node and edge store was retained unchanged
    # in count, and verify the shape/ranges of the newly-added relation.
    for node_type, expected_count in original_node_counts.items():
        if node_count(graph, node_type) != expected_count:
            raise ValueError(
                f"Existing node count changed for {node_type!r}."
            )

    for edge_type, expected_count in original_edge_counts.items():
        if edge_count(graph, edge_type) != expected_count:
            raise ValueError(
                f"Existing edge count changed for {edge_type}."
            )

    new_edges = graph[NEW_EDGE_TYPE].edge_index
    if tuple(new_edges.shape) != (2, row_count):
        raise ValueError("Unexpected new edge-index shape.")
    if int(new_edges.min()) < 0:
        raise ValueError("Negative index in new edge store.")
    if int(new_edges[0].max()) >= len(sample_ids):
        raise ValueError("Sample endpoint outside the sample-node range.")
    if int(new_edges[1].min()) != 0:
        raise ValueError("Observation indices do not start at zero.")
    if int(new_edges[1].max()) != row_count - 1:
        raise ValueError("Observation endpoint range is incomplete.")

    manifest = {
        "schema_version": "3.0.0",
        "purpose": (
            "Graph V2 plus the validated full intergenic observation "
            "dataset; a new artifact, with no changes to Graph V2."
        ),
        "input_graph": str(GRAPH_V2),
        "source_evidence_extract": str(SOURCE),
        "source_evidence_sha256": source_hash_before,
        "output_graph": str(GRAPH_OUT),
        "source_rows": row_count,
        "source_sample_count": len(sample_seen),
        "sample_nodes_matched": len(sample_seen),
        "new_node_type": NEW_NODE_TYPE,
        "new_node_count": row_count,
        "new_edge_type": list(NEW_EDGE_TYPE),
        "new_edge_direction": "sample -> intergenic_observation",
        "new_edge_count": row_count,
        "node_identity": (
            "Unique source_maf_data_row integer, referring to the "
            "validated MAF extract's source row locator. The display "
            "identifier can be derived as "
            "'openpedcan_v15_maf_row:<source_maf_data_row>'."
        ),
        "coordinate_reference": "GRCh38",
        "coordinate_convention": "MAF positions, 1-based inclusive",
        "filtered_records": {
            "Hugo_Symbol": "Unknown",
            "Variant_Classification": "IGR",
            "NCBI_Build": "GRCh38",
        },
        "chromosome_code_map": chromosome_codes,
        "variant_type_code_map": variant_type_codes,
        "original_node_counts_preserved": original_node_counts,
        "original_edge_counts_preserved": {
            "::".join(edge_type): count
            for edge_type, count in original_edge_counts.items()
        },
        "limitations": [
            "Intergenic observations are not assigned to genes.",
            "Chromosome and variant-type codes are category encodings, "
            "not biological scores.",
            "Raw alleles and the remaining extract fields are preserved "
            "in the companion validated source evidence extract.",
        ],
        "validation": {
            "all_rows_streamed_twice": True,
            "expected_row_count": True,
            "all_rows_grch38_igr_unknown": True,
            "all_sample_ids_found_in_graph_v2": True,
            "source_row_ids_strictly_increasing": True,
            "coordinates_valid": True,
            "source_unchanged_during_build": True,
            "old_node_counts_preserved": True,
            "old_edge_counts_preserved": True,
            "new_edges_in_range": True,
        },
    }

    TEMP_MANIFEST.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    torch.save(graph, TEMP_GRAPH)

    # Publish the separately named artifact only after saving succeeds.
    os.replace(TEMP_GRAPH, GRAPH_OUT)
    os.replace(TEMP_MANIFEST, MANIFEST_OUT)

    print("\nFULL GRAPH V3 BUILD")
    print("=" * 50)
    print(f"Existing node types preserved: {len(original_node_counts)}")
    print(f"Existing edge types preserved: {len(original_edge_counts)}")
    print(f"Intergenic observation nodes added: {row_count:,}")
    print(f"Sample-to-observation edges added: {row_count:,}")
    print(f"Distinct matched samples: {len(sample_seen):,}")
    print(f"Graph output: {GRAPH_OUT}")
    print(f"Manifest: {MANIFEST_OUT}")
    print("Original Graph V2 was not modified.")
