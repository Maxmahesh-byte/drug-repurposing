import csv
import json
from collections import Counter
from pathlib import Path

source = Path(
    "data/processed/graph_v2/intergenic_variants.csv"
)
metadata = Path(
    "data/raw/openpedcan/openpedcan_phase1.csv"
)
output = Path(
    "data/processed/graph_v2/intergenic_validation_report.json"
)

EXPECTED_ROWS = 4_522_696

required = [
    "source_data_row",
    "sample_id",
    "gene_symbol",
    "variant_classification",
    "variant_type",
    "chromosome",
    "start_position",
    "reference_allele",
    "tumor_allele",
]

for path in (source, metadata):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

# Load sample identifiers from the OpenPedCan metadata.
metadata_samples = set()

with metadata.open(
    "r", newline="", encoding="utf-8-sig"
) as handle:
    reader = csv.DictReader(handle)

    if "sample_id" not in (reader.fieldnames or []):
        raise ValueError("phase1 metadata has no sample_id column")

    for row in reader:
        sample = (row.get("sample_id") or "").strip()
        if sample:
            metadata_samples.add(sample)

counts = Counter()
variant_types = Counter()
chromosomes = Counter()
seen_source_rows = set()
duplicate_source_rows = 0
invalid_source_rows = 0
invalid_positions = 0

with source.open(
    "r", newline="", encoding="utf-8-sig"
) as handle:
    reader = csv.DictReader(handle)

    missing_columns = set(required) - set(reader.fieldnames or [])
    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    for row in reader:
        counts["records"] += 1

        # Confirm the expected classification and gene label.
        if (row["variant_classification"] or "").strip().upper() == "IGR":
            counts["IGR_records"] += 1

        if (row["gene_symbol"] or "").strip().casefold() == "unknown":
            counts["unknown_gene_label_records"] += 1

        # Check source-row locator.
        locator = (row["source_data_row"] or "").strip()
        try:
            locator_int = int(locator)
            if locator_int < 1:
                raise ValueError
            if locator_int in seen_source_rows:
                duplicate_source_rows += 1
            else:
                seen_source_rows.add(locator_int)
        except ValueError:
            invalid_source_rows += 1

        # Missing-value checks.
        for field in (
            "sample_id",
            "chromosome",
            "start_position",
            "reference_allele",
            "tumor_allele",
        ):
            if not (row.get(field) or "").strip():
                counts[f"missing_{field}"] += 1

        # Check that coordinates contain a positive integer.
        position = (row.get("start_position") or "").strip()
        try:
            if int(position) <= 0:
                raise ValueError
        except ValueError:
            invalid_positions += 1

        # Check sample metadata linkage.
        sample = (row.get("sample_id") or "").strip()
        if not sample:
            counts["records_without_sample_id"] += 1
        elif sample not in metadata_samples:
            counts["records_with_sample_absent_from_phase1"] += 1

        variant_types[
            (row.get("variant_type") or "").strip() or "<missing>"
        ] += 1
        chromosomes[
            (row.get("chromosome") or "").strip() or "<missing>"
        ] += 1

report = {
    "source_file": str(source),
    "metadata_file": str(metadata),
    "expected_records": EXPECTED_ROWS,
    "actual_records": counts["records"],
    "record_count_matches_expected": counts["records"] == EXPECTED_ROWS,
    "IGR_records": counts["IGR_records"],
    "unknown_gene_label_records": counts["unknown_gene_label_records"],
    "distinct_metadata_sample_ids": len(metadata_samples),
    "records_with_sample_absent_from_phase1":
        counts["records_with_sample_absent_from_phase1"],
    "records_without_sample_id": counts["records_without_sample_id"],
    "invalid_or_nonpositive_positions": invalid_positions,
    "invalid_source_row_locators": invalid_source_rows,
    "duplicate_source_row_locators": duplicate_source_rows,
    "missing_fields": {
        field: counts[f"missing_{field}"]
        for field in (
            "sample_id",
            "chromosome",
            "start_position",
            "reference_allele",
            "tumor_allele",
        )
    },
    "variant_type_counts": dict(variant_types.most_common()),
    "top_20_chromosome_labels": dict(chromosomes.most_common(20)),
    "interpretation_note": (
        "Sample absence, missing coordinates, or invalid positions "
        "are audit findings to investigate, not automatic proof "
        "that a record is biologically invalid. Genome assembly and "
        "coordinate conventions still need to be confirmed."
    ),
}

output.write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8",
)

print("INTERGENIC VALIDATION")
print("=" * 55)
print(f"Records checked: {counts['records']:,}")
print(f"Expected count matches: {counts['records'] == EXPECTED_ROWS}")
print(f"IGR records: {counts['IGR_records']:,}")
print(f"Unknown gene labels: {counts['unknown_gene_label_records']:,}")
print(f"Metadata sample IDs: {len(metadata_samples):,}")
print(
    "Records whose sample is absent from phase1: "
    f"{counts['records_with_sample_absent_from_phase1']:,}"
)
print(f"Invalid/nonpositive positions: {invalid_positions:,}")
print(f"Invalid source-row locators: {invalid_source_rows:,}")
print(f"Duplicate source-row locators: {duplicate_source_rows:,}")
print("\nMissing required-field values:")
for field in (
    "sample_id",
    "chromosome",
    "start_position",
    "reference_allele",
    "tumor_allele",
):
    print(f"  {field}: {counts[f'missing_{field}']:,}")

print(f"\nFull report: {output}")
print("No raw files or graphs were modified.")
