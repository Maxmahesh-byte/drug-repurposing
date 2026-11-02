#!/usr/bin/env python3

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "data/raw/gdsc/gdsc2_responses.csv"
OUT = ROOT / "data/ai/baselines"

N_SPLITS = 5
MIN_DRUG_CANCER_N = 10
TARGET = "ln_ic50"

def median(values):
    return float(statistics.median(values))

def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    error = predicted - actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    sst = float(np.sum((actual - actual.mean()) ** 2))
    sse = float(np.sum(error ** 2))

    r2 = None if sst == 0 else float(1 - sse / sst)

    return {
        "n": int(len(actual)),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": None if r2 is None else round(r2, 6),
        "mean_prediction_error": round(float(np.mean(error)), 6),
    }

def main():
    if not SOURCE.exists():
        raise SystemExit(f"Missing source: {SOURCE}")

    rows = []
    result_ids = Counter()
    pair_keys = set()
    missing_cancer = 0

    required = {
        "curve_id", "result_id", "drug_id", "drug_name",
        "cell_line", "sanger_model_id",
        "reported_cancer_type", "ln_ic50",
    }

    with SOURCE.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        absent = required - set(reader.fieldnames or [])
        if absent:
            raise SystemExit(f"Missing columns: {sorted(absent)}")

        for line_no, row in enumerate(reader, start=2):
            curve_id = (row.get("curve_id") or "").strip()
            result_id = (row.get("result_id") or "").strip()
            drug_id = (row.get("drug_id") or "").strip()
            cell_line = (row.get("cell_line") or "").strip()
            sanger_id = (row.get("sanger_model_id") or "").strip()
            cancer = (row.get("reported_cancer_type") or "").strip()

            try:
                y = float(row[TARGET])
            except (TypeError, ValueError):
                raise SystemExit(
                    f"Invalid LN(IC50) at source line {line_no}"
                )

            if not all([curve_id, drug_id, cell_line, sanger_id]):
                raise SystemExit(
                    f"Missing identity field at source line {line_no}"
                )

            if not math.isfinite(y):
                raise SystemExit(
                    f"Non-finite target at source line {line_no}"
                )

            pair = (drug_id, sanger_id)
            if pair in pair_keys:
                raise SystemExit(
                    f"Duplicate drug/Sanger pair detected: {pair}. "
                    "Review replicates before training."
                )
            pair_keys.add(pair)

            result_ids[result_id] += 1
            if not cancer:
                missing_cancer += 1

            rows.append({
                "curve_id": curve_id,
                "drug_id": drug_id,
                "drug_name": (row.get("drug_name") or "").strip(),
                "cell_line": cell_line,
                "sanger_model_id": sanger_id,
                "cancer_type": cancer,
                "actual": y,
            })

    if not rows:
        raise SystemExit("No response records found.")

    groups = np.array(
        [r["sanger_model_id"] for r in rows],
        dtype=object,
    )
    y = np.array([r["actual"] for r in rows], dtype=float)
    drug_ids = [r["drug_id"] for r in rows]
    cancers = [r["cancer_type"] for r in rows]

    unique_groups = set(groups.tolist())
    if len(unique_groups) < N_SPLITS:
        raise SystemExit("Not enough distinct cell lines for five folds.")

    predictions = {
        "B0_global_median": np.full(len(rows), np.nan),
        "B1_drug_median": np.full(len(rows), np.nan),
        "B2_drug_cancer_median": np.full(len(rows), np.nan),
    }

    fold_assignment = np.full(len(rows), -1, dtype=int)
    fold_summary = []

    splitter = GroupKFold(n_splits=N_SPLITS)

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(rows)), y, groups),
        start=1,
    ):
        train_groups = set(groups[train_idx].tolist())
        test_groups = set(groups[test_idx].tolist())

        if train_groups.intersection(test_groups):
            raise RuntimeError(
                f"Cell-line leakage in fold {fold}"
            )

        global_median = median(y[train_idx])

        drug_values = defaultdict(list)
        drug_cancer_values = defaultdict(list)

        for idx in train_idx:
            drug = drug_ids[idx]
            cancer = cancers[idx]

            drug_values[drug].append(y[idx])

            if cancer:
                drug_cancer_values[(drug, cancer)].append(y[idx])

        drug_medians = {
            drug: median(values)
            for drug, values in drug_values.items()
        }

        drug_cancer_medians = {
            key: median(values)
            for key, values in drug_cancer_values.items()
            if len(values) >= MIN_DRUG_CANCER_N
        }

        fallback_count = 0

        for idx in test_idx:
            drug = drug_ids[idx]
            cancer = cancers[idx]

            b0 = global_median
            b1 = drug_medians.get(drug, global_median)

            key = (drug, cancer) if cancer else None

            if key and key in drug_cancer_medians:
                b2 = drug_cancer_medians[key]
            else:
                b2 = b1
                fallback_count += 1

            predictions["B0_global_median"][idx] = b0
            predictions["B1_drug_median"][idx] = b1
            predictions["B2_drug_cancer_median"][idx] = b2

        fold_assignment[test_idx] = fold

        fold_summary.append({
            "fold": fold,
            "training_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "training_cell_lines": int(len(train_groups)),
            "test_cell_lines": int(len(test_groups)),
            "cell_line_overlap": 0,
            "B2_fallback_test_rows": fallback_count,
        })

        print(
            f"Fold {fold}: "
            f"{len(train_idx):,} training rows | "
            f"{len(test_idx):,} test rows | "
            f"{len(test_groups):,} held-out cell lines"
        )

    if np.any(fold_assignment < 1):
        raise RuntimeError("Some records did not receive a test fold.")

    for name, values in predictions.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"Incomplete predictions for {name}")

    # Overall metrics
    overall = {
        name: metrics(y, pred)
        for name, pred in predictions.items()
    }

    # Macro-average MAE: each drug contributes equally.
    macro_drug = {}

    unique_drugs = sorted(set(drug_ids))

    for name, pred in predictions.items():
        drug_maes = []

        for drug in unique_drugs:
            indices = [
                i for i, value in enumerate(drug_ids)
                if value == drug
            ]

            if indices:
                drug_maes.append(
                    float(np.mean(np.abs(y[indices] - pred[indices])))
                )

        macro_drug[name] = round(
            float(np.mean(drug_maes)), 6
        )

    # Per-drug performance report
    per_drug = []

    for drug in unique_drugs:
        indices = [
            i for i, value in enumerate(drug_ids)
            if value == drug
        ]

        entry = {
            "drug_id": drug,
            "drug_name": rows[indices[0]]["drug_name"],
            "test_measurements": len(indices),
        }

        for name, pred in predictions.items():
            entry[name + "_mae"] = round(
                float(np.mean(np.abs(y[indices] - pred[indices]))),
                6,
            )

        per_drug.append(entry)

    OUT.mkdir(parents=True, exist_ok=True)

    prediction_path = OUT / "gdsc2_ln_ic50_oof_predictions.csv"

    with prediction_path.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        fields = [
            "curve_id", "drug_id", "drug_name",
            "cell_line", "sanger_model_id",
            "reported_cancer_type", "fold",
            "actual_ln_ic50",
            *predictions.keys(),
        ]

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i, row in enumerate(rows):
            record = {
                "curve_id": row["curve_id"],
                "drug_id": row["drug_id"],
                "drug_name": row["drug_name"],
                "cell_line": row["cell_line"],
                "sanger_model_id": row["sanger_model_id"],
                "reported_cancer_type": row["cancer_type"],
                "fold": int(fold_assignment[i]),
                "actual_ln_ic50": row["actual"],
            }

            for name, pred in predictions.items():
                record[name] = float(pred[i])

            writer.writerow(record)

    with (OUT / "gdsc2_ln_ic50_per_drug.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(per_drug[0].keys()),
        )
        writer.writeheader()
        writer.writerows(per_drug)

    report = {
        "benchmark": "GDSC2 LN(IC50) baseline",
        "source": str(SOURCE.relative_to(ROOT)),
        "target": TARGET,
        "rows": len(rows),
        "unique_cell_lines": len(unique_groups),
        "unique_drugs": len(unique_drugs),
        "unique_curve_ids": len(set(r["curve_id"] for r in rows)),
        "unique_nonblank_result_ids": len(
            [key for key in result_ids if key]
        ),
        "missing_cancer_type_rows": missing_cancer,
        "split": {
            "method": "GroupKFold",
            "groups": "sanger_model_id",
            "n_splits": N_SPLITS,
            "cell_line_overlap": 0,
        },
        "baseline_definitions": {
            "B0_global_median": "Training-fold global median.",
            "B1_drug_median": "Training-fold median per drug ID; global fallback.",
            "B2_drug_cancer_median": (
                f"Training-fold median per drug ID and reported cancer type, "
                f"minimum group size {MIN_DRUG_CANCER_N}; otherwise B1 fallback."
            ),
        },
        "overall_metrics": overall,
        "macro_average_drug_mae": macro_drug,
        "folds": fold_summary,
        "result_id_warning": (
            "Only one distinct nonblank result_id was found in the source. "
            "curve_id was used as the measurement identifier."
        ),
        "feature_policy": (
            "These are descriptive baselines. They use drug ID and reported "
            "cancer-type metadata only. LN(IC50) is the held-out target, "
            "never an input feature. No molecular features are used."
        ),
        "generalisation_scope": (
            "Unseen cell lines for drugs represented in training. "
            "Not unseen-drug or clinical-response validation."
        ),
    }

    (OUT / "gdsc2_ln_ic50_baseline_metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n========== BASELINE RESULTS ==========")

    for name, result in overall.items():
        print(
            f"{name}: "
            f"MAE={result['mae']:.5f} | "
            f"RMSE={result['rmse']:.5f} | "
            f"R2={result['r2']}"
        )

    print("\nMacro-average per-drug MAE:")
    for name, value in macro_drug.items():
        print(f"  {name}: {value:.5f}")

    print("\nLeakage checks:")
    print("  Held-out cell-line overlap: 0")
    print("  All rows have exactly one out-of-fold prediction: PASS")
    print("  Target used as an input feature: NO")

    print(f"\nOutputs saved in: {OUT}")
    print("Original source data unchanged.")

if __name__ == "__main__":
    main()
