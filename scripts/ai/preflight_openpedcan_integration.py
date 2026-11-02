#!/usr/bin/env python3
"""Read-only preflight for pediatric molecular evidence integration."""

import csv
import gzip
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
OUT = ROOT / "data/ai/audits/openpedcan_integration_preflight.json"

SOURCES = [
    "data/raw/openpedcan/openpedcan_phase1.csv",
    "data/raw/openpedcan/openpedcan_mutations_phase2.csv",
    "data/raw/openpedcan/openpedcan_copy_number_phase3.csv",
    "data/raw/openpedcan/openpedcan_fusions_phase4.csv",
    "data/raw/openpedcan/v15/histologies.tsv",
    "data/raw/openpedcan/v15/consensus_wgs_plus_cnvkit_wxs_plus_freec_tumor_only.tsv.gz",
    "data/raw/openpedcan/v15/fusion-putative-oncogenic.tsv",
    "data/raw/openpedcan/v15/snv-consensus-plus-hotspots.maf.tsv.gz",
]

KEY_HINTS = (
    "sample_id", "biospecimen_id", "tumor_sample_barcode",
    "sample", "aliquot_id", "patient", "kids_first_biospecimen_id",
)
DIAGNOSIS_HINTS = (
    "tumor_type", "diagnosis", "pathology_diagnosis",
    "integrated_diagnosis", "harmonized_diagnosis",
    "broad_histology", "short_histology", "molecular_subtype",
)
MOLECULAR_HINTS = (
    "gene_symbol", "hugo_symbol", "ensembl", "variant_classification",
    "variant_type", "fusion_name", "fusionname", "copy_number",
    "reference_allele", "tumor_allele", "protein_change",
)

def read_header(path):
    opener = gzip.open if path.name.endswith(".gz") else open
    delimiter = "\t" if path.name.endswith((".tsv", ".tsv.gz")) else ","
    with opener(path, "rt", encoding="utf-8-sig", errors="replace") as f:
        return next(csv.reader(f, delimiter=delimiter), [])

def main():
    if not (ROOT / "data").exists():
        raise SystemExit("Run this script from the project root.")

    disk = shutil.disk_usage(ROOT)
    records = []

    for relative in SOURCES:
        path = ROOT / relative

        if not path.exists():
            records.append({
                "path": relative,
                "exists": False,
                "status": "MISSING",
            })
            continue

        try:
            columns = [str(c).strip() for c in read_header(path)]
            records.append({
                "path": relative,
                "exists": True,
                "size_bytes": path.stat().st_size,
                "header_columns": len(columns),
                "columns": columns,
                "candidate_sample_or_biospecimen_keys": [
                    c for c in columns
                    if any(k in c.casefold() for k in KEY_HINTS)
                ],
                "diagnosis_fields": [
                    c for c in columns
                    if any(k in c.casefold() for k in DIAGNOSIS_HINTS)
                ],
                "molecular_fields": [
                    c for c in columns
                    if any(k in c.casefold() for k in MOLECULAR_HINTS)
                ],
                "status": "HEADER_READ_ONLY",
            })
        except Exception as exc:
            records.append({
                "path": relative,
                "exists": True,
                "status": f"HEADER_ERROR: {type(exc).__name__}: {exc}",
            })

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "disk": {
            "free_bytes": disk.free,
            "free_gib": round(disk.free / 1024**3, 2),
        },
        "sources": records,
        "limitations": [
            "No data rows were scanned.",
            "Candidate identifier fields are not proof of equivalence.",
            "Phase exports and versioned source files may overlap.",
            "No sample-to-cell-line mapping or biological inference was made.",
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("OPENPEDCAN INTEGRATION PREFLIGHT")
    print(f"Free disk space: {report['disk']['free_gib']} GiB")

    for item in records:
        print(f"\n{item['path']}")
        print("  Status:", item["status"])
        if item.get("exists") and "size_bytes" in item:
            print(f"  File size: {item['size_bytes'] / 1024**3:.2f} GiB")
            print("  Candidate IDs:",
                  item["candidate_sample_or_biospecimen_keys"])
            print("  Diagnosis fields:", item["diagnosis_fields"])
            print("  Molecular fields:", item["molecular_fields"])

    print(f"\nReport: {OUT.relative_to(ROOT)}")
    print("Read-only preflight complete.")

if __name__ == "__main__":
    main()
