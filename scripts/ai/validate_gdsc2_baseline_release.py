#!/usr/bin/env python3
"""Validate and freeze the existing GDSC2 baseline benchmark."""

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/raw/gdsc/gdsc2_responses.csv"
OUT = ROOT / "data/ai/baselines"

ENDPOINTS = {
    "ln_ic50": {
        "predictions": OUT / "gdsc2_ln_ic50_oof_predictions.csv",
        "metrics": OUT / "gdsc2_ln_ic50_baseline_metrics.json",
        "actual_column": "actual_ln_ic50",
    },
    "auc": {
        "predictions": OUT / "gdsc2_auc_oof_predictions.csv",
        "metrics": OUT / "gdsc2_auc_baseline_metrics.json",
        "actual_column": "actual_auc",
    },
}

MODELS = [
    "B0_global_median",
    "B1_drug_median",
    "B2_drug_cancer_median",
]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def num(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite value: {value}")
    return result

def close(a, b, tol=1e-5):
    return abs(float(a) - float(b)) <= tol

def fail_if(condition, message, errors):
    if condition:
        errors.append(message)

def metrics(actual, predicted):
    n = len(actual)
    if n == 0:
        raise ValueError("Cannot evaluate empty predictions.")

    mae = sum(abs(p - y) for y, p in zip(actual, predicted)) / n
    rmse = math.sqrt(
        sum((p - y) ** 2 for y, p in zip(actual, predicted)) / n
    )

    mean_y = sum(actual) / n
    sst = sum((y - mean_y) ** 2 for y in actual)
    sse = sum((p - y) ** 2 for y, p in zip(actual, predicted))
    r2 = None if sst == 0 else 1 - sse / sst

    return {"n": n, "mae": mae, "rmse": rmse, "r2": r2}

def expected_predictions(rows, folds, target_name):
    """Recalculate each prediction using only other folds."""
    expected = {model: [None] * len(rows) for model in MODELS}

    for fold in sorted(set(folds)):
        train = [i for i, f in enumerate(folds) if f != fold]
        test = [i for i, f in enumerate(folds) if f == fold]

        global_median = statistics.median(
            rows[i][target_name] for i in train
        )

        by_drug = defaultdict(list)
        by_drug_cancer = defaultdict(list)

        for i in train:
            row = rows[i]
            by_drug[row["drug_id"]].append(row[target_name])

            if row["cancer"]:
                by_drug_cancer[
                    (row["drug_id"], row["cancer"])
                ].append(row[target_name])

        drug_median = {
            key: statistics.median(values)
            for key, values in by_drug.items()
        }

        drug_cancer_median = {
            key: statistics.median(values)
            for key, values in by_drug_cancer.items()
            if len(values) >= 10
        }

        for i in test:
            row = rows[i]
            b1 = drug_median.get(row["drug_id"], global_median)
            key = (row["drug_id"], row["cancer"])

            b2 = drug_cancer_median.get(key, b1) if row["cancer"] else b1

            expected["B0_global_median"][i] = global_median
            expected["B1_drug_median"][i] = b1
            expected["B2_drug_cancer_median"][i] = b2

    return expected

def main():
    errors = []
    checks = []

    required_paths = [SOURCE]
    for spec in ENDPOINTS.values():
        required_paths.extend([spec["predictions"], spec["metrics"]])

    for path in required_paths:
        if not path.exists():
            errors.append(f"Missing required file: {path.relative_to(ROOT)}")

    if errors:
        raise SystemExit("\n".join(errors))

    source_rows = read_csv(SOURCE)
    source_by_curve = {}

    for row in source_rows:
        curve = row["curve_id"].strip()
        if not curve or curve in source_by_curve:
            errors.append(f"Missing or duplicate source curve_id: {curve!r}")
            continue

        try:
            source_by_curve[curve] = {
                "drug_id": row["drug_id"].strip(),
                "drug_name": row["drug_name"].strip(),
                "cell_line": row["cell_line"].strip(),
                "sanger_model_id": row["sanger_model_id"].strip(),
                "cancer": row["reported_cancer_type"].strip(),
                "ln_ic50": num(row["ln_ic50"]),
                "auc": num(row["auc"]),
            }
        except (ValueError, KeyError) as exc:
            errors.append(f"Invalid source row for curve {curve}: {exc}")

    if errors:
        raise SystemExit("\n".join(errors))

    loaded = {}

    for endpoint, spec in ENDPOINTS.items():
        pred_rows = read_csv(spec["predictions"])
        metric_report = json.loads(
            spec["metrics"].read_text(encoding="utf-8")
        )
        loaded[endpoint] = (pred_rows, metric_report)

        pred_by_curve = {}
        for row in pred_rows:
            curve = row["curve_id"].strip()
            if not curve or curve in pred_by_curve:
                errors.append(
                    f"{endpoint}: missing or duplicate prediction curve_id {curve!r}"
                )
                continue
            pred_by_curve[curve] = row

        if set(pred_by_curve) != set(source_by_curve):
            errors.append(
                f"{endpoint}: prediction curve IDs do not exactly match source IDs "
                f"(missing={len(set(source_by_curve) - set(pred_by_curve))}, "
                f"extra={len(set(pred_by_curve) - set(source_by_curve))})"
            )
            continue

        ordered_rows = []
        folds = []
        actual = []
        predictions = {model: [] for model in MODELS}

        for source in source_rows:
            curve = source["curve_id"].strip()
            source_record = source_by_curve[curve]
            row = pred_by_curve[curve]

            for field in [
                "drug_id", "cell_line", "sanger_model_id",
                "reported_cancer_type",
            ]:
                expected_value = (
                    source_record["cancer"]
                    if field == "reported_cancer_type"
                    else source_record[field]
                )
                if row[field].strip() != expected_value:
                    errors.append(
                        f"{endpoint}: {field} mismatch for curve {curve}"
                    )

            y = num(row[spec["actual_column"]])
            if not close(y, source_record[endpoint], tol=1e-8):
                errors.append(
                    f"{endpoint}: actual target mismatch for curve {curve}"
                )

            folds.append(int(row["fold"]))
            actual.append(y)
            ordered_rows.append({
                **source_record,
                endpoint: y,
            })

            for model in MODELS:
                predictions[model].append(num(row[model]))

        # Each cell line must be assigned to exactly one held-out fold.
        group_folds = defaultdict(set)
        for row, fold in zip(ordered_rows, folds):
            group_folds[row["sanger_model_id"]].add(fold)

        violations = [
            group for group, assigned in group_folds.items()
            if len(assigned) != 1
        ]
        if violations:
            errors.append(
                f"{endpoint}: {len(violations)} cell lines appear in multiple folds"
            )

        checks.append({
            "name": f"{endpoint}_source_ids_and_targets_match",
            "passed": True,
            "records": len(pred_rows),
        })

        # Recalculate expected predictions solely from training-fold labels.
        expected = expected_predictions(ordered_rows, folds, endpoint)

        for model in MODELS:
            for index, (actual_pred, expected_pred) in enumerate(
                zip(predictions[model], expected[model])
            ):
                if expected_pred is None or not close(
                    actual_pred, expected_pred, tol=1e-7
                ):
                    errors.append(
                        f"{endpoint}: {model} does not reproduce the "
                        f"training-fold median at record index {index}"
                    )
                    break

        checks.append({
            "name": f"{endpoint}_predictions_recomputed_from_training_folds",
            "passed": True,
            "details": "B0/B1/B2 regenerated from training folds only.",
        })

        # Verify the stored overall metrics are reproducible.
        stored = metric_report["overall_metrics"]
        recomputed = {
            model: metrics(actual, predictions[model])
            for model in MODELS
        }

        for model in MODELS:
            for metric_name in ["mae", "rmse", "r2"]:
                stored_value = stored[model].get(metric_name)
                computed_value = recomputed[model][metric_name]

                if stored_value is None or computed_value is None:
                    if stored_value != computed_value:
                        errors.append(
                            f"{endpoint}: {model} {metric_name} mismatch"
                        )
                elif not close(stored_value, computed_value):
                    errors.append(
                        f"{endpoint}: {model} {metric_name} mismatch "
                        f"(stored={stored_value}, recomputed={computed_value})"
                    )

        checks.append({
            "name": f"{endpoint}_metrics_recomputed",
            "passed": True,
            "metrics": recomputed,
        })

        loaded[endpoint] = {
            "folds_by_curve": {
                row["curve_id"]: int(row["fold"])
                for row in pred_rows
            },
            "metrics": metric_report,
        }

    # Ensure the endpoint predictions use the same held-out partitions.
    if all(
        isinstance(loaded[e], dict) and "folds_by_curve" in loaded[e]
        for e in ENDPOINTS
    ):
        ln_folds = loaded["ln_ic50"]["folds_by_curve"]
        auc_folds = loaded["auc"]["folds_by_curve"]

        if ln_folds != auc_folds:
            errors.append("LN(IC50) and AUC fold assignments differ.")
        else:
            checks.append({
                "name": "endpoint_fold_assignments_match",
                "passed": True,
            })

    # Avoid a misleading PASS if any check failed midway.
    report = {
        "benchmark_release": "gdsc2_baselines_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE.relative_to(ROOT)),
        "source_rows": len(source_rows),
        "status": "PASS" if not errors else "FAIL",
        "checks": checks,
        "errors": errors,
        "interpretation": (
            "Validated descriptive baselines for held-out cell lines and "
            "known drugs. This does not establish molecular-model performance, "
            "unseen-drug generalisation or clinical efficacy."
        ),
    }

    report_path = OUT / "gdsc2_baseline_release_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if errors:
        print("BASELINE RELEASE VALIDATION: FAIL")
        for error in errors:
            print(" -", error)
        print(f"Report: {report_path}")
        raise SystemExit(1)

    tracked = [
        SOURCE,
        *required_paths[1:],
        ROOT / "scripts/ai/audit_gdsc2_baseline.py",
        ROOT / "scripts/ai/run_gdsc2_baselines.py",
        ROOT / "scripts/ai/analyze_gdsc2_baselines.py",
        ROOT / "scripts/ai/run_gdsc2_auc_baselines.py",
        ROOT / "scripts/ai/validate_gdsc2_baseline_release.py",
    ]

    files = {}
    for path in tracked:
        if path.exists():
            files[str(path.relative_to(ROOT))] = {
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }

    manifest = {
        **report,
        "files": files,
        "target_policy": {
            "ln_ic50": "Target column only; never a feature.",
            "auc": "Target column only; never a feature.",
        },
        "split_policy": {
            "method": "GroupKFold",
            "group": "sanger_model_id",
            "folds": 5,
            "same_folds_for_ln_ic50_and_auc": True,
        },
        "baseline_models": {
            "B0": "Training-fold global median.",
            "B1": "Training-fold drug median; global fallback.",
            "B2": (
                "Training-fold drug + reported cancer-type median "
                "with minimum group size 10; otherwise B1 fallback."
            ),
        },
    }

    manifest_path = OUT / "gdsc2_baseline_release_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\n========== BASELINE RELEASE VALIDATION ==========")
    print("Status: PASS")
    print(f"Source rows: {len(source_rows):,}")
    for check in checks:
        print(f"  PASS  {check['name']}")
    print(f"\nValidation report: {report_path}")
    print(f"Release manifest:  {manifest_path}")
    print("Source data and baseline predictions unchanged.")

if __name__ == "__main__":
    main()
