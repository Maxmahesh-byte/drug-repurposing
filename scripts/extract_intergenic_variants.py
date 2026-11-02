import csv
import json
from pathlib import Path

source = Path(
    "data/raw/openpedcan/openpedcan_mutations_phase2.csv"
)
output = Path(
    "data/processed/graph_v2/intergenic_variants.csv"
)
manifest = Path(
    "data/processed/graph_v2/intergenic_variants_manifest.json"
)

if not source.exists():
    raise FileNotFoundError(f"Source file not found: {source}")

output.parent.mkdir(parents=True, exist_ok=True)

scanned = 0
extracted = 0

with source.open("r", newline="", encoding="utf-8-sig") as src:
    reader = csv.DictReader(src)

    required = {
        "sample_id",
        "gene_symbol",
        "variant_classification",
        "variant_type",
        "chromosome",
        "start_position",
        "reference_allele",
        "tumor_allele",
    }

    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    # Keep every original source column and add a row locator.
    fields = ["source_data_row"] + list(reader.fieldnames)

    with output.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()

        for row_number, row in enumerate(reader, start=1):
            scanned += 1

            gene = (row.get("gene_symbol") or "").strip().casefold()
            classification = (
                row.get("variant_classification") or ""
            ).strip().upper()

            if gene == "unknown" and classification == "IGR":
                writer.writerow({
                    "source_data_row": row_number,
                    **row,
                })
                extracted += 1

manifest_data = {
    "schema_version": "1.0.0",
    "source_file": str(source),
    "output_file": str(output),
    "source_data_rows_scanned": scanned,
    "records_extracted": extracted,
    "filter": {
        "gene_symbol_casefold": "unknown",
        "variant_classification_uppercase": "IGR",
    },
    "source_data_row_note": (
        "1-based data-row number, excluding the CSV header; "
        "valid only while the source file remains unchanged."
    ),
    "preservation_note": (
        "All original source columns are retained. The extracted "
        "records are observations, not gene assignments."
    ),
}

manifest.write_text(
    json.dumps(manifest_data, indent=2) + "\n",
    encoding="utf-8",
)

print("INTERGENIC RECORD EXTRACTION")
print("=" * 50)
print(f"Source rows scanned: {scanned:,}")
print(f"Records extracted:   {extracted:,}")
print(f"Output:              {output}")
print(f"Manifest:            {manifest}")

if extracted != 4_522_696:
    print(
        "WARNING: extracted count differs from the earlier audit. "
        "Investigate the source version or filter before proceeding."
    )
else:
    print("Count matches the earlier audit.")
