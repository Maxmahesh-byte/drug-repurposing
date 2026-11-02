#!/usr/bin/env python3
"""Read-only coverage audit for canonical OpenPedCan molecular exports."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
OUT = ROOT / "data/ai/audits/openpedcan_sample_coverage.json"

def load_ids(path, column, delimiter=","):
    values = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter=delimiter):
            value = (row.get(column) or "").strip()
            if value:
                values.add(value)
    return values

clinical_path = ROOT / "data/raw/openpedcan/openpedcan_phase1.csv"
histology_path = ROOT / "data/raw/openpedcan/v15/histologies.tsv"

clinical_ids = load_ids(clinical_path, "sample_id")
histology_ids = load_ids(histology_path, "sample_id", "\t")

sources = [
    ("mutations", "data/raw/openpedcan/openpedcan_mutations_phase2.csv",
     "sample_id", "gene_symbol"),
    ("copy_number", "data/raw/openpedcan/openpedcan_copy_number_phase3.csv",
     "sample_id", "gene_symbol"),
    ("fusions", "data/raw/openpedcan/openpedcan_fusions_phase4.csv",
     "sample_id", "fusion_name"),
]

results = []

for name, relative, sample_col, feature_col in sources:
    path = ROOT / relative
    if not path.exists():
        results.append({"layer": name, "status": "MISSING", "path": relative})
        continue

    rows = empty_ids = empty_features = 0
    unique_ids = set()
    clinical_matched_rows = histology_matched_rows = 0
    missing_from_both_rows = 0

    print(f"Scanning {name}: {relative}", flush=True)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {sample_col, feature_col}
        absent = required - set(reader.fieldnames or [])

        if absent:
            results.append({
                "layer": name,
                "status": "MISSING_REQUIRED_COLUMNS",
                "columns_missing": sorted(absent),
                "path": relative,
            })
            continue

        for row in reader:
            rows += 1
            sample = (row.get(sample_col) or "").strip()
            feature = (row.get(feature_col) or "").strip()

            if not sample:
                empty_ids += 1
                continue

            unique_ids.add(sample)
            if not feature:
                empty_features += 1

            in_clinical = sample in clinical_ids
            in_histology = sample in histology_ids

            if in_clinical:
                clinical_matched_rows += 1
            if in_histology:
                histology_matched_rows += 1
            if not in_clinical and not in_histology:
                missing_from_both_rows += 1

            if rows % 2_000_000 == 0:
                print(f"  {rows:,} rows scanned", flush=True)

    matched_ids_clinical = unique_ids & clinical_ids
    matched_ids_histology = unique_ids & histology_ids

    result = {
        "layer": name,
        "path": relative,
        "status": "SCANNED",
        "rows": rows,
        "unique_nonblank_sample_ids": len(unique_ids),
        "rows_with_blank_sample_id": empty_ids,
        "rows_with_blank_feature": empty_features,
        "unique_sample_ids_in_clinical": len(matched_ids_clinical),
        "unique_sample_ids_in_histology": len(matched_ids_histology),
        "unique_sample_ids_in_neither": len(
            unique_ids - clinical_ids - histology_ids
        ),
        "rows_matching_clinical_sample_ids": clinical_matched_rows,
        "rows_matching_histology_sample_ids": histology_matched_rows,
        "rows_matching_neither": missing_from_both_rows,
    }
    results.append(result)

    print(
        f"  Rows: {rows:,}; unique sample IDs: {len(unique_ids):,}; "
        f"IDs in neither metadata table: "
        f"{result['unique_sample_ids_in_neither']:,}"
    )

report = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "clinical_unique_sample_ids": len(clinical_ids),
    "histology_unique_sample_ids": len(histology_ids),
    "layers": results,
    "limitations": [
        "Exact string matching only.",
        "Matching a sample identifier does not validate biological equivalence.",
        "A sample absent from these metadata tables requires review; it is not automatically invalid.",
        "No records were joined or modified.",
    ],
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(f"\nReport: {OUT.relative_to(ROOT)}")
print("Read-only coverage audit complete.")
