#!/usr/bin/env python3

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "data/raw/gdsc/gdsc2_responses.csv"
OUTPUT = ROOT / "data/ai/audits/gdsc2_baseline_audit.json"

def clean(value):
    return str(value or "").strip()

def parse_number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None

def percentiles(values):
    if not values:
        return {}
    ordered = sorted(values)

    def percentile(p):
        position = (len(ordered) - 1) * p
        low = int(position)
        high = min(low + 1, len(ordered))
        fraction = position - low
        return ordered[low] * (1 - fraction) + ordered[high] * fraction

    return {
        "min": round(ordered[0], 5),
        "p01": round(percentile(0.01), 5),
        "p25": round(percentile(0.25), 5),
        "median": round(percentile(0.50), 5),
        "p75": round(percentile(0.75), 5),
        "p99": round(percentile(0.99), 5),
        "max": round(ordered[-1], 5),
        "mean": round(statistics.fmean(ordered), 5),
    }

def main():
    if not SOURCE.exists():
        raise SystemExit(f"Missing source file: {SOURCE}")

    row_count = 0
    missing = Counter()
    unique = defaultdict(set)

    target_values = {
        "ln_ic50": [],
        "auc": [],
    }

    pair_stats = defaultdict(
        lambda: {
            "records": 0,
            "ln_ic50_min": None,
            "ln_ic50_max": None,
            "auc_min": None,
            "auc_max": None,
        }
    )

    model_names = defaultdict(set)
    model_cancer_types = defaultdict(set)
    name_model_ids = defaultdict(set)

    result_ids = Counter()
    curve_ids = Counter()
    datasets = Counter()

    with SOURCE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)

        required = {
            "drug_id", "drug_name", "cell_line",
            "ln_ic50", "auc",
        }

        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise SystemExit(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        for row in reader:
            row_count += 1

            drug_id = clean(row.get("drug_id"))
            drug_name = clean(row.get("drug_name"))
            cell_line = clean(row.get("cell_line"))
            sanger_id = clean(row.get("sanger_model_id"))
            cosmic_id = clean(row.get("cosmic_id"))
            cancer_type = clean(row.get("reported_cancer_type"))

            ln_ic50 = parse_number(row.get("ln_ic50"))
            auc = parse_number(row.get("auc"))

            for field, value in {
                "drug_id": drug_id,
                "drug_name": drug_name,
                "cell_line": cell_line,
                "sanger_model_id": sanger_id,
                "cosmic_id": cosmic_id,
                "reported_cancer_type": cancer_type,
                "ln_ic50": ln_ic50,
                "auc": auc,
            }.items():
                if value in ("", None):
                    missing[field] += 1

            for field, value in {
                "drug_id": drug_id,
                "drug_name": drug_name,
                "cell_line": cell_line,
                "sanger_model_id": sanger_id,
                "cosmic_id": cosmic_id,
            }.items():
                if value not in ("", None):
                    unique[field].add(value)

            result_id = clean(row.get("result_id"))
            curve_id = clean(row.get("curve_id"))
            dataset = clean(row.get("dataset"))

            if result_id:
                result_ids[result_id] += 1
            if curve_id:
                curve_ids[curve_id] += 1
            if dataset:
                datasets[dataset] += 1

            if sanger_id:
                group_id = f"SANGER:{sanger_id}"
                if cell_line:
                    model_names[sanger_id].add(cell_line)
                if cancer_type:
                    model_cancer_types[sanger_id].add(cancer_type)
            elif cell_line:
                group_id = f"NAME:{cell_line.casefold()}"
            else:
                group_id = None

            if cell_line and sanger_id:
                name_model_ids[cell_line.casefold()].add(sanger_id)

            if not drug_id or not group_id:
                continue

            pair = pair_stats[(drug_id, group_id)]
            pair["records"] += 1

            for label, value in [
                ("ln_ic50", ln_ic50),
                ("auc", auc),
            ]:
                if value is None:
                    continue

                target_values[label].append(value)
                low_key = f"{label}_min"
                high_key = f"{label}_max"

                if pair[low_key] is None or value < pair[low_key]:
                    pair[low_key] = value
                if pair[high_key] is None or value > pair[high_key]:
                    pair[high_key] = value

    repeated = [
        item for item in pair_stats.values()
        if item["records"] > 1
    ]

    ln_variable = [
        item for item in repeated
        if item["ln_ic50_min"] is not None
        and item["ln_ic50_max"] is not None
        and item["ln_ic50_max"] - item["ln_ic50_min"] > 1e-10
    ]

    auc_variable = [
        item for item in repeated
        if item["auc_min"] is not None
        and item["auc_max"] is not None
        and item["auc_max"] - item["auc_min"] > 1e-10
    ]

    report = {
        "source": str(SOURCE.relative_to(ROOT)),
        "rows": row_count,
        "unique_identifiers": {
            key: len(value)
            for key, value in unique.items()
        },
        "unique_drug_cell_line_pairs": len(pair_stats),
        "repeated_pairs": len(repeated),
        "repeated_pairs_with_variable_ln_ic50": len(ln_variable),
        "repeated_pairs_with_variable_auc": len(auc_variable),
        "maximum_records_per_pair": max(
            (item["records"] for item in pair_stats.values()),
            default=0,
        ),
        "missing_values": dict(missing),
        "targets": {
            key: {
                "valid_values": len(value),
                "distribution": percentiles(value),
            }
            for key, value in target_values.items()
        },
        "identifier_quality": {
            "result_id_unique_count": len(result_ids),
            "result_id_repeated_values": sum(
                1 for count in result_ids.values() if count > 1
            ),
            "curve_id_unique_count": len(curve_ids),
            "curve_id_repeated_values": sum(
                1 for count in curve_ids.values() if count > 1
            ),
            "cell_line_names_with_multiple_sanger_ids": sum(
                1 for ids in name_model_ids.values() if len(ids) > 1
            ),
            "sanger_ids_with_multiple_cell_line_names": sum(
                1 for names in model_names.values() if len(names) > 1
            ),
            "sanger_ids_with_multiple_cancer_labels": sum(
                1 for labels in model_cancer_types.values()
                if len(labels) > 1
            ),
        },
        "dataset_row_counts": dict(datasets),
        "five_fold_group_split_possible": (
            len(unique["sanger_model_id"]) >= 5
        ),
        "interpretation": [
            "Repeated drug-cell-line pairs need review before aggregation.",
            "LN(IC50) and AUC are separate prediction targets.",
            "Response measurements must never be model input features.",
            "Cell-line identity consistency must be checked before splitting.",
            "This audit does not train a model or establish predictive performance.",
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n========== GDSC2 BASELINE DATA AUDIT ==========")
    print(f"Rows: {row_count:,}")

    print("\nUnique identifiers:")
    for key, value in report["unique_identifiers"].items():
        print(f"  {key}: {value:,}")

    print(f"\nDrug-cell-line pairs: {report['unique_drug_cell_line_pairs']:,}")
    print(f"Repeated pairs: {report['repeated_pairs']:,}")
    print(f"Repeated pairs with variable LN(IC50): "
          f"{report['repeated_pairs_with_variable_ln_ic50']:,}")
    print(f"Repeated pairs with variable AUC: "
          f"{report['repeated_pairs_with_variable_auc']:,}")

    print("\nMissing values:")
    for key, value in missing.items():
        print(f"  {key}: {value:,}")

    print("\nTarget distributions:")
    for key, value in report["targets"].items():
        print(f"\n  {key}: {value['valid_values']:,} valid values")
        for metric, number in value["distribution"].items():
            print(f"    {metric}: {number}")

    print("\nIdentifier quality:")
    for key, value in report["identifier_quality"].items():
        print(f"  {key}: {value:,}")

    print("\nFive-fold group split possible:",
          report["five_fold_group_split_possible"])

    print(f"\nAudit report saved: {OUTPUT}")
    print("Source data unchanged. No model trained.")

if __name__ == "__main__":
    main()
