import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone

source = Path(
    "data/raw/openpedcan/v15/snv-consensus-plus-hotspots.maf.tsv.gz"
)
extracted = Path(
    "data/processed/graph_v2/intergenic_variants.csv"
)
report_path = Path(
    "data/processed/graph_v2/intergenic_maf_build_audit.json"
)
reference_path = Path(
    "data/processed/graph_v2/intergenic_coordinate_reference.json"
)

EXPECTED_INTERGENIC = 4_522_696

for path in (source, extracted, reference_path):
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")

build_counts = Counter()
intergenic_build_counts = Counter()
total_rows = 0
intergenic_rows = 0
missing_build_rows = 0

with gzip.open(source, "rt", newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, delimiter="\t")

    required = {
        "NCBI_Build",
        "Hugo_Symbol",
        "Variant_Classification",
    }
    missing = required - set(reader.fieldnames or [])

    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")

    for row in reader:
        total_rows += 1
        build = (row.get("NCBI_Build") or "").strip() or "<missing>"
        build_counts[build] += 1

        if build == "<missing>":
            missing_build_rows += 1

        gene = (row.get("Hugo_Symbol") or "").strip().casefold()
        classification = (
            row.get("Variant_Classification") or ""
        ).strip().upper()

        if gene == "unknown" and classification == "IGR":
            intergenic_rows += 1
            intergenic_build_counts[build] += 1

with extracted.open(
    "r", newline="", encoding="utf-8-sig"
) as handle:
    extracted_count = sum(1 for _ in csv.DictReader(handle))

matches_expected = intergenic_rows == EXPECTED_INTERGENIC
matches_extracted = intergenic_rows == extracted_count
all_intergenic_grch38 = (
    intergenic_build_counts.get("GRCh38", 0) == intergenic_rows
    and len(intergenic_build_counts) == 1
)

report = {
    "source_file": str(source),
    "extracted_file": str(extracted),
    "source_rows_scanned": total_rows,
    "source_build_counts": dict(build_counts),
    "intergenic_unknown_rows_in_source": intergenic_rows,
    "intergenic_build_counts": dict(intergenic_build_counts),
    "extracted_intergenic_rows": extracted_count,
    "intergenic_count_matches_expected": matches_expected,
    "source_intergenic_count_matches_extraction": matches_extracted,
    "all_intergenic_rows_are_grch38": all_intergenic_grch38,
    "missing_build_rows": missing_build_rows,
    "audited_at_utc": datetime.now(timezone.utc).isoformat(),
}

report_path.write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8",
)

print("SOURCE GENOME BUILD AUDIT")
print("=" * 55)
print(f"Source rows scanned: {total_rows:,}")
print(f"Build counts: {dict(build_counts)}")
print(f"Intergenic source rows: {intergenic_rows:,}")
print(f"Intergenic build counts: {dict(intergenic_build_counts)}")
print(f"Extracted rows: {extracted_count:,}")
print(f"Expected count matches: {matches_expected}")
print(f"Source count matches extraction: {matches_extracted}")
print(f"All intergenic rows GRCh38: {all_intergenic_grch38}")
print(f"Rows missing build: {missing_build_rows:,}")
print(f"\nReport: {report_path}")

# Update the sidecar only when the audit fully supports it.
if matches_expected and matches_extracted and all_intergenic_grch38:
    reference = json.loads(
        reference_path.read_text(encoding="utf-8")
    )
    reference["genome_assembly"] = "GRCh38"
    reference["assembly_status"] = (
        "Verified against the NCBI_Build field for every matching "
        "Unknown/IGR row in the original compressed source file."
    )
    reference["coordinate_system"] = (
        "Mutation Annotation Format Start_Position and End_Position "
        "are 1-based and inclusive, per the GDC MAF specification."
    )
    reference["build_audit_report"] = str(report_path)
    reference["validation_status"] = (
        "Extraction count and source build verified; coordinate "
        "convention documented from the MAF specification."
    )
    reference_path.write_text(
        json.dumps(reference, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Coordinate-reference sidecar updated after successful audit.")
else:
    print(
        "Sidecar unchanged: investigate the audit report before "
        "marking the assembly verified."
    )

print("Raw data and Graph V2 were not modified.")
