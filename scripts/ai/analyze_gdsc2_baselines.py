#!/usr/bin/env python3

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/ai/baselines/gdsc2_ln_ic50_oof_predictions.csv"
OUT = ROOT / "data/ai/baselines/gdsc2_baseline_diagnostic_report.json"

MODELS = [
    "B0_global_median",
    "B1_drug_median",
    "B2_drug_cancer_median",
]

def mae(actual, predicted):
    return float(np.mean(np.abs(actual - predicted)))

def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))

def r2(actual, predicted):
    denominator = np.sum((actual - np.mean(actual)) ** 2)
    if denominator == 0:
        return None
    return float(1 - np.sum((actual - predicted) ** 2) / denominator)

def main():
    if not SOURCE.exists():
        raise SystemExit(f"Missing predictions file: {SOURCE}")

    rows = []

    with SOURCE.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "curve_id": row["curve_id"],
                "drug_id": row["drug_id"],
                "drug_name": row["drug_name"],
                "cell_line": row["cell_line"],
                "sanger_model_id": row["sanger_model_id"],
                "cancer_type": row["reported_cancer_type"],
                "fold": int(row["fold"]),
                "actual": float(row["actual_ln_ic50"]),
                **{
                    model: float(row[model])
                    for model in MODELS
                },
            })

    if not rows:
        raise SystemExit("No OOF predictions found.")

    actual = np.array([r["actual"] for r in rows])
    groups = np.array([r["sanger_model_id"] for r in rows])
    drug_ids = np.array([r["drug_id"] for r in rows])
    folds = np.array([r["fold"] for r in rows])

    predictions = {
        model: np.array([r[model] for r in rows])
        for model in MODELS
    }

    # Confirm each cell line belongs to exactly one held-out fold.
    group_fold_counts = defaultdict(set)
    for row in rows:
        group_fold_counts[row["sanger_model_id"]].add(row["fold"])

    split_violations = [
        group for group, assigned_folds in group_fold_counts.items()
        if len(assigned_folds) != 1
    ]

    if split_violations:
        raise SystemExit(
            f"Cell-line fold assignment violation: {len(split_violations)}"
        )

    # Overall metrics.
    overall = {}
    for model in MODELS:
        pred = predictions[model]
        overall[model] = {
            "n": len(rows),
            "mae": mae(actual, pred),
            "rmse": rmse(actual, pred),
            "r2": r2(actual, pred),
        }

    # Fold-level metrics.
    fold_report = []
    for fold in sorted(set(folds)):
        mask = folds == fold
        fold_result = {
            "fold": int(fold),
            "test_rows": int(mask.sum()),
            "test_cell_lines": int(len(set(groups[mask]))),
            "metrics": {},
        }

        for model in MODELS:
            fold_result["metrics"][model] = {
                "mae": mae(actual[mask], predictions[model][mask]),
                "rmse": rmse(actual[mask], predictions[model][mask]),
                "r2": r2(actual[mask], predictions[model][mask]),
            }

        train_mask = folds != fold
        train_drugs = set(drug_ids[train_mask])
        test_drugs = set(drug_ids[mask])
        unseen = test_drugs - train_drugs

        fold_result["unseen_test_drugs"] = sorted(unseen)
        fold_result["test_rows_for_unseen_drugs"] = int(
            sum(
                1 for i in np.where(mask)[0]
                if drug_ids[i] in unseen
            )
        )
        fold_report.append(fold_result)

    # Per-drug comparison.
    per_drug = []
    by_drug = defaultdict(list)

    for i, row in enumerate(rows):
        by_drug[row["drug_id"]].append(i)

    b2_better = 0
    b2_equal = 0
    b2_worse = 0

    for drug, indices in sorted(by_drug.items()):
        idx = np.array(indices)
        b1_error = np.abs(actual[idx] - predictions["B1_drug_median"][idx])
        b2_error = np.abs(actual[idx] - predictions["B2_drug_cancer_median"][idx])

        b1_mae = float(np.mean(b1_error))
        b2_mae = float(np.mean(b2_error))
        delta = b1_mae - b2_mae

        if delta > 1e-10:
            b2_better += 1
        elif delta < -1e-10:
            b2_worse += 1
        else:
            b2_equal += 1

        per_drug.append({
            "drug_id": drug,
            "drug_name": rows[indices[0]]["drug_name"],
            "n": len(indices),
            "B1_mae": b1_mae,
            "B2_mae": b2_mae,
            "B1_minus_B2_mae": delta,
        })

    per_drug.sort(key=lambda x: x["B1_minus_B2_mae"], reverse=True)

    # Paired cluster bootstrap: resample whole cell lines.
    unique_groups = np.array(sorted(set(groups)))
    group_to_indices = {
        group: np.where(groups == group)[0]
        for group in unique_groups
    }

    cluster_stats = np.zeros((len(unique_groups), 3), dtype=float)

    for gi, group in enumerate(unique_groups):
        idx = group_to_indices[group]
        cluster_stats[gi, 0] = np.abs(
            actual[idx] - predictions["B1_drug_median"][idx]
        ).sum()
        cluster_stats[gi, 1] = np.abs(
            actual[idx] - predictions["B2_drug_cancer_median"][idx]
        ).sum()
        cluster_stats[gi, 2] = len(idx)

    rng = np.random.default_rng(20261027)
    n_boot = 2000
    n_groups = len(unique_groups)

    sampled = rng.integers(
        0, n_groups, size=(n_boot, n_groups)
    )
    totals = cluster_stats[sampled].sum(axis=1)

    b1_boot = totals[:, 0] / totals[:, 2]
    b2_boot = totals[:, 1] / totals[:, 2]
    delta_boot = b1_boot - b2_boot

    def ci(values):
        return {
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }

    delta = (
        overall["B1_drug_median"]["mae"]
        - overall["B2_drug_cancer_median"]["mae"]
    )

    report = {
        "source": str(SOURCE.relative_to(ROOT)),
        "rows": len(rows),
        "cell_lines": len(unique_groups),
        "drugs": len(by_drug),
        "split_integrity": {
            "each_cell_line_in_exactly_one_test_fold": True,
            "violations": len(split_violations),
        },
        "overall_metrics": overall,
        "fold_metrics": fold_report,
        "B2_vs_B1": {
            "absolute_MAE_improvement": delta,
            "relative_MAE_improvement_percent": (
                100 * delta / overall["B1_drug_median"]["mae"]
            ),
            "cluster_bootstrap_95_percent_CI_for_MAE_improvement": ci(
                delta_boot
            ),
            "B2_better_drugs": b2_better,
            "B2_equal_drugs": b2_equal,
            "B2_worse_drugs": b2_worse,
            "drug_count": len(by_drug),
        },
        "bootstrap": {
            "method": "Paired resampling of whole Sanger cell-line groups",
            "replicates": n_boot,
            "seed": 20261027,
        },
        "interpretation": (
            "This evaluates known-drug response estimation on held-out "
            "cell lines. It does not evaluate unseen-drug generalisation "
            "or paediatric clinical response."
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n========== BASELINE DIAGNOSTICS ==========")
    print(f"Rows: {len(rows):,}")
    print(f"Cell lines: {len(unique_groups):,}")
    print(f"Drugs: {len(by_drug):,}")
    print("Cell-line split integrity: PASS")

    print("\nFold-level MAE:")
    for fold in fold_report:
        parts = [
            f"{name}={fold['metrics'][name]['mae']:.5f}"
            for name in MODELS
        ]
        print(f"  Fold {fold['fold']}: " + " | ".join(parts))
        print(
            f"    Unseen drugs in test: "
            f"{len(fold['unseen_test_drugs'])}"
        )

    comparison = report["B2_vs_B1"]
    print("\nB2 versus B1:")
    print(
        "  MAE improvement: "
        f"{comparison['absolute_MAE_improvement']:.5f}"
    )
    print(
        "  Relative improvement: "
        f"{comparison['relative_MAE_improvement_percent']:.2f}%"
    )
    print(
        "  Cluster-bootstrap 95% CI: "
        f"{comparison['cluster_bootstrap_95_percent_CI_for_MAE_improvement']}"
    )
    print(
        "  Drugs: B2 better / equal / worse = "
        f"{b2_better} / {b2_equal} / {b2_worse}"
    )

    print(f"\nReport saved: {OUT}")
    print("No source data or predictions modified.")

if __name__ == "__main__":
    main()
