from collections import Counter
from pathlib import Path
import pandas as pd
import csv

source = Path("data/raw/openpedcan/openpedcan_mutations_phase2.csv")
output = Path(
    "data/processed/graph_v2/intergenic_variant_summary.csv"
)

if not source.exists():
    raise FileNotFoundError(source)

output.parent.mkdir(parents=True, exist_ok=True)

unknown_labels = {
    "", "unknown", "na", "n/a", "nan", "null", "none", "."
}

counts = Counter()
total_rows = 0
unassigned_rows = 0
classification_counts = Counter()

for chunk in pd.read_csv(
    source,
    usecols=[
        "sample_id",
        "gene_symbol",
        "variant_classification",
        "variant_type",
    ],
    dtype="string",
    chunksize=250_000,
):
    total_rows += len(chunk)

    symbol = chunk["gene_symbol"].str.strip()
    normalized = symbol.str.casefold()
    mask = symbol.isna() | normalized.isin(unknown_labels)

    unassigned = chunk.loc[mask].copy()
    unassigned_rows += len(unassigned)

    if unassigned.empty:
        continue

    unassigned["sample_id"] = (
        unassigned["sample_id"]
        .fillna("<MISSING_SAMPLE>")
        .str.strip()
        .replace("", "<MISSING_SAMPLE>")
    )

    for column in ["variant_classification", "variant_type"]:
        unassigned[column] = (
            unassigned[column]
            .fillna("<MISSING>")
            .str.strip()
            .replace("", "<BLANK>")
        )

    classification_counts.update(
        unassigned["variant_classification"].value_counts().to_dict()
    )

    grouped = unassigned.groupby(
        ["sample_id", "variant_classification", "variant_type"],
        dropna=False,
    ).size()

    for key, value in grouped.items():
        counts[key] += int(value)

with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "sample_id",
        "variant_classification",
        "variant_type",
        "raw_record_count",
        "source_file",
    ])

    for (sample, classification, variant_type), count in sorted(counts.items()):
        writer.writerow([
            sample,
            classification,
            variant_type,
            count,
            str(source),
        ])

print("INTERGENIC / UNASSIGNED VARIANT SUMMARY")
print("=" * 60)
print(f"Source rows scanned: {total_rows:,}")
print(f"Rows with unassigned gene symbols: {unassigned_rows:,}")
print(f"Summary groups written: {len(counts):,}")
print(f"Summary record counts total: {sum(counts.values()):,}")
print("\nClassifications:")
for label, count in classification_counts.most_common():
    print(f"  {label}: {count:,}")
print(f"\nOutput: {output}")
print("Raw data and existing graph were not modified.")
