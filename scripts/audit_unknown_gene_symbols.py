from collections import Counter
from pathlib import Path
import pandas as pd

path = Path("data/raw/openpedcan/openpedcan_mutations_phase2.csv")

if not path.exists():
    raise FileNotFoundError(path)

total_rows = 0
symbol_counts = Counter()
rows_by_sample = Counter()
unknown_by_sample = Counter()
unknown_classifications = Counter()
unknown_variant_types = Counter()

unknown_labels = {
    "", "unknown", "na", "n/a", "nan", "null", "none", "."
}

for chunk in pd.read_csv(
    path,
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

    samples = chunk["sample_id"].fillna("<MISSING_SAMPLE>").str.strip()
    symbols = chunk["gene_symbol"].str.strip()
    normalized = symbols.str.casefold()

    unknown = symbols.isna() | normalized.isin(unknown_labels)

    symbol_labels = symbols.fillna("<MISSING>")
    symbol_labels = symbol_labels.mask(
        symbol_labels.str.strip().eq(""),
        "<BLANK>",
    )
    symbol_counts.update(symbol_labels.value_counts().to_dict())

    rows_by_sample.update(samples.value_counts().to_dict())
    unknown_by_sample.update(
        samples[unknown].value_counts().to_dict()
    )

    unknown_rows = chunk.loc[unknown]

    for column, counter in [
        ("variant_classification", unknown_classifications),
        ("variant_type", unknown_variant_types),
    ]:
        values = (
            unknown_rows[column]
            .fillna("<MISSING>")
            .str.strip()
            .replace("", "<BLANK>")
        )
        counter.update(values.value_counts().to_dict())

unknown_total = sum(unknown_by_sample.values())

print("\nRAW MUTATION FILE SUMMARY")
print("=" * 65)
print(f"Total rows: {total_rows:,}")
print(f"Distinct gene_symbol labels: {len(symbol_counts):,}")
print(
    f"Rows with missing/unknown gene symbols: "
    f"{unknown_total:,} ({unknown_total / max(total_rows, 1):.2%})"
)

print("\nMOST COMMON gene_symbol LABELS")
for label, count in symbol_counts.most_common(20):
    print(f"{label!r}: {count:,}")

print("\nSAMPLES WITH MOST UNKNOWN-GENE RECORDS")
for sample, count in unknown_by_sample.most_common(20):
    total = rows_by_sample[sample]
    print(
        f"{sample}: unknown={count:,}, "
        f"all_records={total:,}, "
        f"unknown_fraction={count / max(total, 1):.2%}"
    )

print("\nVARIANT CLASSIFICATIONS AMONG UNKNOWN-GENE RECORDS")
for label, count in unknown_classifications.most_common(15):
    print(f"{label!r}: {count:,}")

print("\nVARIANT TYPES AMONG UNKNOWN-GENE RECORDS")
for label, count in unknown_variant_types.most_common(15):
    print(f"{label!r}: {count:,}")

print("\nAudit complete. No data files were modified.")
