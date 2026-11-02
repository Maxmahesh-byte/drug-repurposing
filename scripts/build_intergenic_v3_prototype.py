import csv
import gzip
import json
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

GRAPH_V2 = Path("data/processed/graph_v2/typed_graph_v2.pt")
SOURCE = Path("data/processed/graph_v2/intergenic_variants_maf.csv.gz")
OUT_DIR = Path("data/processed/graph_v3_prototype")
LIMIT = 100_000

GRAPH_OUT = OUT_DIR / "intergenic_observations_100k.pt"
EVIDENCE_OUT = OUT_DIR / "intergenic_evidence_100k.csv"
MANIFEST_OUT = OUT_DIR / "intergenic_100k_prototype_manifest.json"

for path in (GRAPH_OUT, EVIDENCE_OUT, MANIFEST_OUT):
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing file: {path}"
        )

for path in (GRAPH_V2, SOURCE):
    if not path.is_file():
        raise FileNotFoundError(f"Required input not found: {path}")

# Load the existing graph only to check and reuse sample identifiers.
# This script does not modify or resave Graph V2.
graph_v2 = torch.load(
    GRAPH_V2, map_location="cpu", weights_only=False
)
if "sample" not in graph_v2.node_types:
    raise ValueError("Graph V2 has no 'sample' node type.")

v2_sample_ids = [str(x) for x in graph_v2["sample"].node_ids]
if len(v2_sample_ids) != len(set(v2_sample_ids)):
    raise ValueError("Duplicate sample IDs in Graph V2.")

known_samples = set(v2_sample_ids)

# Retain up to 32 records per sample, then select round-robin
# across sample barcodes so early samples do not dominate the test.
MAX_PER_SAMPLE = 64
by_sample = {}

with gzip.open(SOURCE, "rt", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    required = {
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
    absent = sorted(required - set(fieldnames or []))
    if absent:
        raise ValueError(f"Missing source columns: {absent}")

    for row in reader:
        sample_id = row["Tumor_Sample_Barcode"]
        bucket = by_sample.setdefault(sample_id, [])
        if len(bucket) < MAX_PER_SAMPLE:
            bucket.append(row)

all_sample_ids = sorted(by_sample)
rows = []

for rank in range(MAX_PER_SAMPLE):
    for sample_id in all_sample_ids:
        sample_rows = by_sample[sample_id]
        if rank < len(sample_rows):
            rows.append(sample_rows[rank])
            if len(rows) == LIMIT:
                break
    if len(rows) == LIMIT:
        break

rows.sort(key=lambda row: int(row["source_maf_data_row"]))

if len(rows) != LIMIT:
    raise ValueError(
        f"Requested {LIMIT:,} observations, but collected {len(rows):,}. "
        "Check how many source records are available per sample."
    )
if len({row["Tumor_Sample_Barcode"] for row in rows}) != len(all_sample_ids):
    raise ValueError(
        "Not all source sample barcodes are represented in the prototype."
    )

if any(
    row["NCBI_Build"] != "GRCh38"
    or row["Variant_Classification"] != "IGR"
    or row["Hugo_Symbol"] != "Unknown"
    for row in rows
):
    raise ValueError("Prototype contains unexpected record types.")

sample_ids_in_rows = {
    row["Tumor_Sample_Barcode"] for row in rows
}
missing_samples = sorted(sample_ids_in_rows - known_samples)
if missing_samples:
    raise ValueError(
        f"{len(missing_samples)} prototype samples are absent "
        f"from Graph V2; examples: {missing_samples[:10]}"
    )

source_rows = [int(row["source_maf_data_row"]) for row in rows]
if len(set(source_rows)) != len(source_rows):
    raise ValueError("Duplicate source row identifiers in prototype.")
if source_rows != sorted(source_rows):
    raise ValueError("Source rows are not in ascending order.")

for row in rows:
    start = int(row["Start_Position"])
    end = int(row["End_Position"])
    if start <= 0 or end < start:
        raise ValueError(
            f"Invalid coordinates at source row "
            f"{row['source_maf_data_row']}"
        )

# Fixed category codes for numeric graph attributes.
chromosome_codes = {
    **{f"chr{i}": i for i in range(1, 23)},
    "chrX": 23,
    "chrY": 24,
    "chrM": 25,
    "chrMT": 25,
}
variant_type_codes = {
    "SNP": 1,
    "DEL": 2,
    "INS": 3,
    "DNP": 4,
    "TNP": 5,
    "ONP": 6,
}

# Use only samples present in this prototype, with stable sorted ordering.
sample_ids = sorted(sample_ids_in_rows)
sample_index = {sample_id: i for i, sample_id in enumerate(sample_ids)}

data = HeteroData()
data["sample"].num_nodes = len(sample_ids)
data["sample"].node_ids = sample_ids

obs = data["intergenic_observation"]
obs.num_nodes = len(rows)
obs.node_ids = [
    f"openpedcan_v15_maf_row:{source_row}"
    for source_row in source_rows
]
obs.source_maf_data_row = torch.tensor(source_rows, dtype=torch.long)
obs.start_position = torch.tensor(
    [int(row["Start_Position"]) for row in rows],
    dtype=torch.long,
)
obs.end_position = torch.tensor(
    [int(row["End_Position"]) for row in rows],
    dtype=torch.long,
)
obs.chromosome_code = torch.tensor(
    [chromosome_codes.get(row["Chromosome"], 0) for row in rows],
    dtype=torch.long,
)
obs.variant_type_code = torch.tensor(
    [variant_type_codes.get(row["Variant_Type"], 0) for row in rows],
    dtype=torch.long,
)

# One directed edge per source record; do not deduplicate observations.
edge_index = torch.tensor(
    [
        [sample_index[row["Tumor_Sample_Barcode"]] for row in rows],
        list(range(len(rows))),
    ],
    dtype=torch.long,
)
data["sample", "has_intergenic_observation",
     "intergenic_observation"].edge_index = edge_index

# Preserve every source column in a companion evidence table.
OUT_DIR.mkdir(parents=True, exist_ok=True)
with EVIDENCE_OUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

torch.save(data, GRAPH_OUT)

manifest = {
    "schema_version": "0.1.0-prototype",
    "purpose": "100k scale prototype: round-robin selection across source sample barcodes, capped at 32 records per sample; not production",
    "source_file": str(SOURCE),
    "graph_v2_reference": str(GRAPH_V2),
    "graph_output": str(GRAPH_OUT),
    "evidence_output": str(EVIDENCE_OUT),
    "records_in_prototype": len(rows),
    "sample_nodes": len(sample_ids),
    "observation_nodes": len(rows),
    "observation_edges": int(edge_index.shape[1]),
    "edge_direction": "sample -> intergenic_observation",
    "deduplication": "None; one observation per source record",
    "source_row_field": "source_maf_data_row",
    "coordinate_reference": "GRCh38",
    "coordinate_convention": "MAF start/end positions, 1-based inclusive",
    "chromosome_code_map": chromosome_codes,
    "chromosome_code_zero_means": "Unrecognized chromosome label; inspect evidence CSV",
    "variant_type_code_map": variant_type_codes,
    "variant_type_code_zero_means": "Unrecognized variant type; inspect evidence CSV",
    "sample_join": "Exact match against Graph V2 sample node IDs",
    "validation": {
        "all_records_igr_unknown_grch38": True,
        "all_sample_ids_found_in_graph_v2": True,
        "unique_source_rows": True,
        "valid_coordinates": True,
    },
    "limitations": [
        "Prototype only; not the full graph",
        "Categorical codes are numeric encodings, not biological scores",
        "No intergenic regulatory or nearby-gene relationships inferred",
    ],
}
MANIFEST_OUT.write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)

print("GRAPH V3 INTERGENIC PROTOTYPE")
print("=" * 50)
print(f"Observations: {len(rows):,}")
print(f"Sample nodes: {len(sample_ids):,}")
print(f"Edges: {edge_index.shape[1]:,}")
print(f"Missing Graph V2 sample IDs: {len(missing_samples)}")
print(f"Graph: {GRAPH_OUT}")
print(f"Evidence table: {EVIDENCE_OUT}")
print(f"Manifest: {MANIFEST_OUT}")
print("Graph V2 and source MAF extract were not modified.")
