#!/usr/bin/env python3

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/raw/gdsc/gdsc2_responses.csv"
FOLDS = ROOT / "data/ai/baselines/gdsc2_ln_ic50_oof_predictions.csv"
OUT = ROOT / "data/ai/baselines"

MODELS = [
    "B0_global_median",
    "B1_drug_median",
    "B2_drug_cancer_median",
]

def med(values):
    return float(statistics.median(values))

def get_metrics(actual, predicted):
    error = predicted - actual
    sst = np.sum((actual - actual.mean()) ** 2)

    return {
        "n": len(actual),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "r2": (
            None if sst == 0
            else float(1 - np.sum(error ** 2) / sst)
        ),
    }

def main():
    # Recover the exact existing LN(IC50) fold assignments.
    fold_by_curve = {}

    with FOLDS.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            curve = row["curve_id"]

            if curve in fold_by_curve:
                raise ValueError(f"Duplicate curve ID in fold file: {curve}")

            fold_by_curve[curve] = int(row["fold"])

    rows = []
    seen_curves = set()

    with SOURCE.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            curve = row["curve_id"].strip()

            if curve in seen_curves:
                raise ValueError(f"Duplicate source curve ID: {curve}")
            seen_curves.add(curve)

            if curve not in fold_by_curve:
                raise ValueError(f"Missing fold assignment: {curve}")

            try:
                auc = float(row["auc"])
            except (TypeError, ValueError):
                raise ValueError(f"Invalid AUC for curve {curve}")

            if not math.isfinite(auc):
                raise ValueError(f"Non-finite AUC for curve {curve}")

            rows.append({
                "curve_id": curve,
                "drug_id": row["drug_id"].strip(),
                "drug_name": row["drug_name"].strip(),
                "cell_line": row["cell_line"].strip(),
                "sanger_model_id": row["sanger_model_id"].strip(),
                "cancer_type": row["reported_cancer_type"].strip(),
                "fold": fold_by_curve[curve],
                "actual": auc,
            })

    if set(fold_by_curve) != seen_curves:
        raise ValueError("Source and LN(IC50) fold curve IDs do not match.")

    if not rows:
        raise ValueError("No AUC records.")

    n = len(rows)
    actual = np.array([r["actual"] for r in rows], dtype=float)
    groups = np.array([r["sanger_model_id"] for r in rows])
    folds = np.array([r["fold"] for r in rows])

    drug_ids = [r["drug_id"] for r in rows]
    cancers = [r["cancer_type"] for r in rows]

    predictions = {
        name: np.full(n, np.nan)
        for name in MODELS
    }

    fold_report = []
    fallback_total = 0

    for fold in sorted(set(folds)):
        train_idx = np.where(folds != fold)[0]
        test_idx = np.where(folds == fold)[0]

        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])

        if train_groups.intersection(test_groups):
            raise RuntimeError("Cell-line leakage detected.")

        global_median = med(actual[train_idx])

        drug_values = defaultdict(list)
        drug_cancer_values = defaultdict(list)

        for i in train_idx:
            drug_values[drug_ids[i]].append(actual[i])

            if cancers[i]:
                drug_cancer_values[
                    (drug_ids[i], cancers[i])
                ].append(actual[i])

        drug_medians = {
            drug: med(values)
            for drug, values in drug_values.items()
        }

        drug_cancer_medians = {
            key: med(values)
            for key, values in drug_cancer_values.items()
            if len(values) >= 10
        }

        fallback = 0

        for i in test_idx:
            drug = drug_ids[i]
            cancer = cancers[i]

            b0 = global_median
            b1 = drug_medians.get(drug, global_median)

            key = (drug, cancer) if cancer else None

            if key and key in drug_cancer_medians:
                b2 = drug_cancer_medians[key]
            else:
                b2 = b1
                fallback += 1

            predictions["B0_global_median"][i] = b0
            predictions["B1_drug_median"][i] = b1
            predictions["B2_drug_cancer_median"][i] = b2

        fallback_total += fallback

        fold_report.append({
            "fold": int(fold),
            "training_rows": len(train_idx),
            "test_rows": len(test_idx),
            "training_cell_lines": len(train_groups),
            "test_cell_lines": len(test_groups),
            "cell_line_overlap": 0,
            "B2_fallback_rows": fallback,
        })

    if any(not np.isfinite(p).all() for p in predictions.values()):
        raise RuntimeError("Missing out-of-fold predictions.")

    overall = {
        name: get_metrics(actual, pred)
        for name, pred in predictions.items()
    }

    unique_drugs = sorted(set(drug_ids))
    per_drug = []

    for drug in unique_drugs:
        indices = np.array([
            i for i, d in enumerate(drug_ids) if d == drug
        ])

        result = {
            "drug_id": drug,
            "drug_name": rows[indices[0]]["drug_name"],
            "n": int(len(indices)),
        }

        for name, pred in predictions.items():
            result[name + "_mae"] = float(
                np.mean(np.abs(actual[indices] - pred[indices]))
            )

        per_drug.append(result)

    macro_mae = {
        name: float(np.mean([
            item[name + "_mae"] for item in per_drug
        ]))
        for name in MODELS
    }

    # Paired bootstrap of whole cell-line clusters.
    unique_groups = np.array(sorted(set(groups)))

    sums = np.zeros((len(unique_groups), 3), dtype=float)

    for gi, group in enumerate(unique_groups):
        idx = np.where(groups == group)[0]

        sums[gi, 0] = np.abs(
            actual[idx] - predictions["B1_drug_median"][idx]
        ).sum()

        sums[gi, 1] = np.abs(
            actual[idx] - predictions["B2_drug_cancer_median"][idx]
        ).sum()

        sums[gi, 2] = len(idx)

    rng = np.random.default_rng(20261028)
    sampled = rng.integers(
        0, len(unique_groups),
        size=(2000, len(unique_groups))
    )
    boot = sums[sampled].sum(axis=1)

    improvements = (
        boot[:, 0] - boot[:, 1]
    ) / boot[:, 2]

    b2_better = sum(
        item["B2_drug_cancer_median_mae"]
        < item["B1_drug_median_mae"] - 1e-10
        for item in per_drug
    )

    b2_worse = sum(
        item["B2_drug_cancer_median_mae"]
        > item["B1_drug_median_mae"] + 1e-10
        for item in per_drug
    )

    report = {
        "benchmark": "GDSC2 AUC baseline",
        "source": str(SOURCE.relative_to(ROOT)),
        "target": "auc",
        "rows": n,
        "cell_lines": len(unique_groups),
        "drugs": len(unique_drugs),
        "folds_reused_from": str(FOLDS.relative_to(ROOT)),
        "split_method": "GroupKFold by Sanger model ID",
        "split_integrity": "PASS",
        "overall_metrics": overall,
        "macro_average_drug_mae": macro_mae,
        "B2_vs_B1": {
            "absolute_mae_improvement": (
                overall["B1_drug_median"]["mae"]
                - overall["B2_drug_cancer_median"]["mae"]
            ),
            "relative_mae_improvement_percent": (
                100 * (
                    overall["B1_drug_median"]["mae"]
                    - overall["B2_drug_cancer_median"]["mae"]
                ) / overall["B1_drug_median"]["mae"]
            ),
            "cluster_bootstrap_95_CI": [
                float(np.quantile(improvements, 0.025)),
                float(np.quantile(improvements, 0.975)),
            ],
            "drugs_B2_better": b2_better,
            "drugs_B2_worse": b2_worse,
        },
        "B2_fallback_rows": fallback_total,
        "folds": fold_report,
        "interpretation": (
            "Experimental AUC estimation on held-out cell lines "
            "for drugs represented in training. Not clinical efficacy."
        ),
    }

    OUT.mkdir(parents=True, exist_ok=True)

    (OUT / "gdsc2_auc_baseline_metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    with (OUT / "gdsc2_auc_oof_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        fields = [
            "curve_id", "drug_id", "drug_name",
            "cell_line", "sanger_model_id",
            "reported_cancer_type", "fold", "actual_auc",
            *MODELS,
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i, row in enumerate(rows):
            output = {
                "curve_id": row["curve_id"],
                "drug_id": row["drug_id"],
                "drug_name": row["drug_name"],
                "cell_line": row["cell_line"],
                "sanger_model_id": row["sanger_model_id"],
                "reported_cancer_type": row["cancer_type"],
                "fold": row["fold"],
                "actual_auc": row["actual"],
            }
            for name, pred in predictions.items():
                output[name] = float(pred[i])
            writer.writerow(output)

    with (OUT / "gdsc2_auc_per_drug.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(per_drug[0].keys())
        )
        writer.writeheader()
        writer.writerows(per_drug)

    print("\n========== GDSC2 AUC BASELINES ==========")
    for name, value in overall.items():
        print(
            f"{name}: MAE={value['mae']:.6f} | "
            f"RMSE={value['rmse']:.6f} | R2={value['r2']}"
        )

    print("\nMacro-average per-drug MAE:")
    for name, value in macro_mae.items():
        print(f"  {name}: {value:.6f}")

    print("\nB2 versus B1:")
    print(json.dumps(report["B2_vs_B1"], indent=2))
    print(f"\nB2 fallback rows: {fallback_total:,}")
    print("Original fold assignments reused: YES")
    print("Target used as an input feature: NO")
    print(f"Report: {OUT / 'gdsc2_auc_baseline_metrics.json'}")

if __name__ == "__main__":
    main()
