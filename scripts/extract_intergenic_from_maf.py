import csv
import gzip
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

source = Path(
    "data/raw/openpedcan/v15/"
    "snv-consensus-plus-hotspots.maf.tsv.gz"
)
output = Path(
    "data/processed/graph_v2/"
    "intergenic_variants_maf.csv.gz"
)
manifest_path = Path(
    "data/processed/graph_v2/"
    "intergenic_variants_maf_manifest.json"
)

EXPECTED_ROWS = 4_522_696

columns = [
    "Hugo_Symbol",
    "Entrez_Gene_Id",
    "NCBI_Build",
    "Chromosome",
    "Start_Position",
    "End_Position",
    "Strand",
    "Variant_Classification",
    "Variant_Type",
    "Reference_Allele",
    "Tumor_Seq_Allele1",
    "Tumor_Seq_Allele2",
    "dbSNP_RS",
    "Tumor_Sample_Barcode",
    "Matched_Norm_Sample_Barcode",
]

if not source.exists():
    raise FileNotFoundError(f"Source file not found: {source}")

output.parent.mkdir(parents=True, exist_ok=True)

scanned = 0
written = 0
build_counts = Counter()
variant_type_counts = Counter()
distinct_samples = set()
missing_required_values = Counter()

with gzip.open(
    source, "rt", newline="", encoding="utf-8"
) as src:
    reader = csv.DictReader(src, delimiter="\t")

    required = {
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
    missing = required - set(reader.fieldnames or [])

    if missing:
        raise ValueError(
            f"Missing MAF columns: {sorted(missing)}"
        )

    with gzip.open(
        output, "wt", newline="", encoding="utf-8"
    ) as dst:
        fields = ["source_maf_data_row"] + columns
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()

        for row_number, row in enumerate(reader, start=1):
            scanned += 1

            gene = (
                row.get("Hugo_Symbol") or ""
            ).strip().casefold()
            classification = (
                row.get("Variant_Classification") or ""
            ).strip().upper()

            if gene != "unknown" or classification != "IGR":
                continue

            written += 1
            build = (row.get("NCBI_Build") or "").strip()
            build_counts[build or "<missing>"] += 1

            variant_type = (
                row.get("Variant_Type") or ""
            ).strip() or "<missing>"
            variant_type_counts[variant_type] += 1

            sample_id = (
                row.get("Tumor_Sample_Barcode") or ""
            ).strip()

            if sample_id:
                distinct_samples.add(sample_id)

            for field in (
                "NCBI_Build",
                "Chromosome",
                "Start_Position",
                "End_Position",
                "Reference_Allele",
                "Tumor_Seq_Allele1",
                "Tumor_Seq_Allele2",
                "Tumor_Sample_Barcode",
            ):
                if not (row.get(field) or "").strip():
                    missing_required_values[field] += 1

            writer.writerow({
                "source_maf_data_row": row_number,
                **{col: row.get(col, "") for col in columns},
            })

report = {
    "schema_version": "1.0.0",
    "source_file": str(source),
    "output_file": str(output),
    "source_data_rows_scanned": scanned,
    "records_written": written,
    "expected_records": EXPECTED_ROWS,
    "count_matches_previous_audits": written == EXPECTED_ROWS,
    "filter": {
        "Hugo_Symbol_casefold": "unknown",
        "Variant_Classification_uppercase": "IGR",
    },
    "source_build_counts": dict(build_counts),
    "distinct_tumor_sample_barcodes": len(distinct_samples),
    "variant_type_counts": dict(variant_type_counts),
    "missing_required_values": dict(missing_required_values),
    "preserved_columns": fields,
    "row_locator_note": (
        "source_maf_data_row is a 1-based data-row number, "
        "excluding the header, in the original compressed MAF."
    ),
    "coordinate_note": (
        "Original MAF start and end positions are retained as text "
        "without coordinate conversion."
    ),
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}

manifest_path.write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8",
)

print("SOURCE-FAITHFUL INTERGENIC EXTRACTION")
print("=" * 55)
print(f"Source rows scanned: {scanned:,}")
print(f"Intergenic rows written: {written:,}")
print(f"Count matches previous audits: {written == EXPECTED_ROWS}")
print(f"Build counts: {dict(build_counts)}")
print(f"Distinct sample barcodes: {len(distinct_samples):,}")
print(f"Missing required values: {dict(missing_required_values)}")
print(f"Output: {output}")
print(f"Manifest: {manifest_path}")
print("Raw data and Graph V2 were not modified.")
