import csv
import gzip
from collections import Counter
from pathlib import Path

path = Path(
    "data/processed/graph_v2/intergenic_variants_maf.csv.gz"
)

rows = 0
builds = Counter()
classifications = Counter()
symbols = Counter()
samples = set()
missing = Counter()
invalid_coordinates = 0
invalid_row_ids = 0
previous_source_row = 0

required = [
    "Hugo_Symbol",
    "NCBI_Build",
    "Chromosome",
    "Start_Position",
    "End_Position",
    "Variant_Classification",
    "Variant_Type",
    "Reference_Allele",
    "Tumor_Sample_Barcode",
]

with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)

    fields = set(reader.fieldnames or [])
    absent_columns = sorted(set(required) - fields)

    for row in reader:
        rows += 1

        for field in required:
            if not (row.get(field) or "").strip():
                missing[field] += 1

        builds[row["NCBI_Build"]] += 1
        classifications[row["Variant_Classification"]] += 1
        symbols[row["Hugo_Symbol"]] += 1
        samples.add(row["Tumor_Sample_Barcode"])

        try:
            start = int(row["Start_Position"])
            end = int(row["End_Position"])
            if start <= 0 or end < start:
                invalid_coordinates += 1
        except (TypeError, ValueError):
            invalid_coordinates += 1

        try:
            source_row = int(row["source_maf_data_row"])
            if source_row <= previous_source_row:
                invalid_row_ids += 1
            previous_source_row = source_row
        except (KeyError, TypeError, ValueError):
            invalid_row_ids += 1

print("INTERGENIC MAF EXTRACT VALIDATION")
print("=" * 50)
print(f"Rows: {rows:,}")
print(f"Missing required columns: {absent_columns}")
print(f"Builds: {dict(builds)}")
print(f"Classifications: {dict(classifications)}")
print(f"Gene symbols: {dict(symbols)}")
print(f"Distinct samples: {len(samples):,}")
print(f"Missing required values: {dict(missing)}")
print(f"Invalid coordinates: {invalid_coordinates:,}")
print(f"Invalid source row order/IDs: {invalid_row_ids:,}")

passed = (
    rows == 4_522_696
    and not absent_columns
    and builds == {"GRCh38": 4_522_696}
    and classifications == {"IGR": 4_522_696}
    and symbols == {"Unknown": 4_522_696}
    and not missing
    and invalid_coordinates == 0
    and invalid_row_ids == 0
)

print(f"\nRESULT: {'PASS' if passed else 'INVESTIGATE'}")
