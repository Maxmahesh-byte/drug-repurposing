#!/usr/bin/env python3
"""Paired cell-line-cluster bootstrap for GDSC2 pair-GNN ablation runs.

The input OOF files contain one prediction per measured drug/cell-line curve.
Uncertainty is estimated by resampling whole Sanger model IDs within each
frozen outer fold, preserving all drug measurements from a sampled cell line.
This avoids treating correlated curves from one cell line as independent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENDPOINTS = ("ln_ic50", "auc")
VARIANTS = ("full_graph", "drug_id_only", "expression_only")


def metric_block(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) == 0 or actual.shape != predicted.shape:
        raise ValueError("Metric inputs must be non-empty arrays with matching shapes.")
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
    }


def cluster_bootstrap_mae_difference(
    frame: pd.DataFrame,
    actual_column: str,
    prediction_a: str,
    prediction_b: str,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate MAE(A)-MAE(B), resampling Sanger model clusters within folds."""
    required = {
        "sanger_model_id", "fold", actual_column, prediction_a, prediction_b
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Bootstrap input is missing columns: {sorted(missing)}")
    if n_bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates.")
    if frame[list(required)].isna().any().any():
        raise ValueError("Bootstrap input contains missing cluster, fold, target, or prediction.")
    if frame.groupby("sanger_model_id")["fold"].nunique().gt(1).any():
        raise ValueError("A Sanger model ID appears in more than one outer fold.")

    actual = frame[actual_column].to_numpy(dtype=float)
    pred_a = frame[prediction_a].to_numpy(dtype=float)
    pred_b = frame[prediction_b].to_numpy(dtype=float)
    if not (
        np.isfinite(actual).all()
        and np.isfinite(pred_a).all()
        and np.isfinite(pred_b).all()
    ):
        raise ValueError("Bootstrap input contains non-finite values.")

    point_delta = float(
        mean_absolute_error(actual, pred_a)
        - mean_absolute_error(actual, pred_b)
    )

    # Pre-aggregate absolute errors by cell line. Each resampled cluster carries
    # every response curve belonging to that Sanger model ID.
    fold_clusters: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _fold, fold_frame in frame.groupby("fold", sort=True):
        cluster_sums = (
            fold_frame.assign(
                _error_a=np.abs(
                    fold_frame[actual_column].to_numpy(dtype=float)
                    - fold_frame[prediction_a].to_numpy(dtype=float)
                ),
                _error_b=np.abs(
                    fold_frame[actual_column].to_numpy(dtype=float)
                    - fold_frame[prediction_b].to_numpy(dtype=float)
                ),
                _rows=1,
            )
            .groupby("sanger_model_id", sort=True)
            .agg(sum_a=("_error_a", "sum"), sum_b=("_error_b", "sum"), rows=("_rows", "sum"))
        )
        if cluster_sums.empty:
            continue
        fold_clusters.append(
            (
                cluster_sums["sum_a"].to_numpy(dtype=float),
                cluster_sums["sum_b"].to_numpy(dtype=float),
                cluster_sums["rows"].to_numpy(dtype=float),
            )
        )
    if not fold_clusters:
        raise ValueError("No Sanger model clusters available for bootstrap.")

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=float)
    for bootstrap_index in range(n_bootstrap):
        total_a = 0.0
        total_b = 0.0
        total_rows = 0.0
        for sums_a, sums_b, counts in fold_clusters:
            sampled = rng.integers(0, len(counts), size=len(counts))
            total_a += float(sums_a[sampled].sum())
            total_b += float(sums_b[sampled].sum())
            total_rows += float(counts[sampled].sum())
        deltas[bootstrap_index] = (total_a - total_b) / total_rows

    lower, upper = np.quantile(deltas, [0.025, 0.975])
    return {
        "point_mae_difference_a_minus_b": point_delta,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "bootstrap_probability_a_has_lower_mae": float(np.mean(deltas < 0)),
        "bootstrap_replicates": int(n_bootstrap),
        "resampling_unit": "Sanger model ID; all drug-response rows kept together",
        "stratification": "resample Sanger model IDs within each frozen outer fold",
        "n_cell_lines": int(frame["sanger_model_id"].nunique()),
        "n_response_rows": int(len(frame)),
    }


def load_run(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Ablation OOF file not found: {path}")
    frame = pd.read_csv(
        path,
        dtype={
            "curve_id": str,
            "drug_id": str,
            "sanger_model_id": str,
        },
    )
    required = {
        "curve_id", "sanger_model_id", "fold",
        "actual_ln_ic50", "actual_auc",
        *[f"pred_{variant}_{endpoint}" for variant in VARIANTS for endpoint in ENDPOINTS],
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required OOF columns: {sorted(missing)}")
    if frame["curve_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate curve IDs.")
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    if set(frame["fold"].unique()) != set(range(1, 6)):
        raise ValueError(f"{path} must contain frozen folds 1–5.")
    if frame["sanger_model_id"].eq("").any():
        raise ValueError(f"{path} contains empty Sanger model IDs.")
    if frame.groupby("sanger_model_id")["fold"].nunique().gt(1).any():
        raise ValueError(f"{path} assigns a Sanger model to multiple folds.")
    return frame.sort_values("curve_id").reset_index(drop=True)


def join_external_comparator(
    run_frame: pd.DataFrame,
    path: Path,
    actual_column: str,
    prediction_column: str,
    endpoint: str,
) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    baseline = pd.read_csv(
        path,
        dtype={"curve_id": str, "drug_id": str, "sanger_model_id": str},
    )
    required = {"curve_id", "fold", actual_column, prediction_column}
    missing = required - set(baseline.columns)
    if missing:
        raise ValueError(f"Comparator {path} is missing columns: {sorted(missing)}")
    if baseline["curve_id"].duplicated().any():
        raise ValueError(f"Comparator {path} contains duplicate curve IDs.")

    actual_name = f"actual_{endpoint}"
    joined = run_frame[
        ["curve_id", "sanger_model_id", "fold", actual_name,
         f"pred_full_graph_{endpoint}"]
    ].merge(
        baseline[["curve_id", "fold", actual_column, prediction_column]].rename(
            columns={
                "fold": "baseline_fold",
                actual_column: "baseline_actual",
                prediction_column: "baseline_prediction",
            }
        ),
        on="curve_id",
        how="inner",
        validate="one_to_one",
    )
    if joined.empty:
        return joined
    if not np.array_equal(joined["fold"].to_numpy(), joined["baseline_fold"].to_numpy()):
        raise ValueError(f"Frozen fold assignments differ in comparator {path}.")
    if not np.allclose(
        joined[actual_name].to_numpy(dtype=float),
        joined["baseline_actual"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-8,
    ):
        raise ValueError(f"Measured {endpoint} values disagree in comparator {path}.")
    return joined.rename(
        columns={
            actual_name: "actual",
            f"pred_full_graph_{endpoint}": "pred_full_graph",
            "baseline_prediction": "pred_comparator",
        }
    )


def main(args) -> dict[str, Any]:
    run_paths = [Path(value).expanduser() for value in args.run]
    run_labels = [path.parent.name or f"run_{index + 1}" for index, path in enumerate(run_paths)]
    if len(run_labels) != len(set(run_labels)):
        run_labels = [f"run_{index + 1}_{label}" for index, label in enumerate(run_labels)]

    baseline_dir = ROOT / "data/ai/baselines"
    molecular_dir = ROOT / "data/ai/molecular_models"
    comparator_specs = {
        "drug_cancer_median_ln_ic50": (
            baseline_dir / "gdsc2_ln_ic50_oof_predictions.csv",
            "actual_ln_ic50", "B2_drug_cancer_median", "ln_ic50",
        ),
        "drug_cancer_median_auc": (
            baseline_dir / "gdsc2_auc_oof_predictions.csv",
            "actual_auc", "B2_drug_cancer_median", "auc",
        ),
        "expression_ridge_ln_ic50": (
            molecular_dir / "gdsc2_expression_ln_ic50_oof_predictions.csv",
            "ln_ic50", "expression_ridge", "ln_ic50",
        ),
        "expression_ridge_auc": (
            molecular_dir / "gdsc2_expression_auc_oof_predictions.csv",
            "auc", "expression_ridge", "auc",
        ),
    }

    metric_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    seed_delta_values: dict[str, list[float]] = {}
    common_curve_ids: set[str] | None = None

    for run_index, (label, run_path) in enumerate(zip(run_labels, run_paths)):
        frame = load_run(run_path)
        curve_ids = set(frame["curve_id"])
        common_curve_ids = curve_ids if common_curve_ids is None else common_curve_ids & curve_ids
        for endpoint in ENDPOINTS:
            actual_column = f"actual_{endpoint}"
            actual = frame[actual_column].to_numpy(dtype=float)
            for variant in VARIANTS:
                prediction_column = f"pred_{variant}_{endpoint}"
                metrics = metric_block(actual, frame[prediction_column].to_numpy(dtype=float))
                metric_rows.append(
                    {
                        "run": label, "endpoint": endpoint, "model": variant, **metrics
                    }
                )

            for comparator_name, comparator_column in (
                ("drug_id_only", f"pred_drug_id_only_{endpoint}"),
                ("expression_only", f"pred_expression_only_{endpoint}"),
            ):
                key = f"{endpoint}:full_graph_vs_{comparator_name}"
                result = cluster_bootstrap_mae_difference(
                    frame,
                    actual_column,
                    f"pred_full_graph_{endpoint}",
                    comparator_column,
                    n_bootstrap=args.bootstrap_replicates,
                    seed=args.seed + run_index * 100 + (0 if endpoint == "ln_ic50" else 1),
                )
                bootstrap_rows.append(
                    {
                        "run": label,
                        "endpoint": endpoint,
                        "comparison": f"full_graph_minus_{comparator_name}",
                        **result,
                    }
                )
                seed_delta_values.setdefault(key, []).append(
                    result["point_mae_difference_a_minus_b"]
                )

            for comparator_name, (path, baseline_actual, baseline_prediction, baseline_endpoint) in comparator_specs.items():
                if baseline_endpoint != endpoint:
                    continue
                joined = join_external_comparator(
                    frame, path, baseline_actual, baseline_prediction, endpoint
                )
                if joined is None or joined.empty:
                    continue
                result = cluster_bootstrap_mae_difference(
                    joined.rename(
                        columns={
                            "actual": actual_column,
                            "pred_full_graph": f"pred_full_graph_{endpoint}",
                            "pred_comparator": "pred_external_baseline",
                        }
                    ),
                    actual_column,
                    f"pred_full_graph_{endpoint}",
                    "pred_external_baseline",
                    n_bootstrap=args.bootstrap_replicates,
                    seed=args.seed + run_index * 1000 + len(bootstrap_rows),
                )
                bootstrap_rows.append(
                    {
                        "run": label,
                        "endpoint": endpoint,
                        "comparison": f"full_graph_minus_{comparator_name}",
                        **result,
                    }
                )
                key = f"{endpoint}:full_graph_vs_{comparator_name}"
                seed_delta_values.setdefault(key, []).append(
                    result["point_mae_difference_a_minus_b"]
                )

    metric_frame = pd.DataFrame(metric_rows)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    across_seeds: dict[str, dict] = {}
    for key, values in seed_delta_values.items():
        array = np.asarray(values, dtype=float)
        across_seeds[key] = {
            "runs": int(len(array)),
            "mean_mae_difference": float(array.mean()),
            "sd_across_runs": float(array.std(ddof=1)) if len(array) > 1 else None,
            "run_differences": array.tolist(),
        }

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_path = output_dir / "gdsc2_pair_gnn_uncertainty_metrics.csv"
    bootstrap_path = output_dir / "gdsc2_pair_gnn_cluster_bootstrap.csv"
    summary_path = output_dir / "gdsc2_pair_gnn_uncertainty_summary.json"
    metric_frame.to_csv(metric_path, index=False)
    bootstrap_frame.to_csv(bootstrap_path, index=False)

    summary = {
        "status": "completed",
        "experiment": "paired Sanger-model-cluster bootstrap for GDSC2 ablation OOF predictions",
        "runs": [
            {"label": label, "path": str(path), "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()}
            for label, path in zip(run_labels, run_paths)
        ],
        "run_count": len(run_paths),
        "shared_curve_id_count_across_runs": (
            len(common_curve_ids) if common_curve_ids is not None else 0
        ),
        "bootstrap": {
            "replicates_per_comparison_per_run": args.bootstrap_replicates,
            "unit": "Sanger model ID; all measured drug-response curves for each sampled cell line remain together",
            "stratification": "resample cell-line clusters within each frozen outer fold",
            "paired": True,
            "interpretation": "For delta MAE = full_graph - comparator, negative values favour full_graph; a 95% interval spanning zero is inconclusive.",
        },
        "per_run_metrics": metric_rows,
        "per_run_cluster_bootstrap": bootstrap_rows,
        "across_run_descriptive_differences": across_seeds,
        "outputs": {
            "metrics_csv": str(metric_path),
            "cluster_bootstrap_csv": str(bootstrap_path),
            "summary_json": str(summary_path),
        },
        "limitations": [
            "Bootstrap intervals quantify held-out cell-line sampling variability conditional on each trained run; they do not by themselves account for all model-training randomness.",
            "Repeated runs with different seeds are needed to assess sensitivity to initialization and inner validation splits.",
            "All results are known-drug in-vitro cell-line benchmarks, not patient outcomes or clinical response predictions.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "run_count": len(run_paths),
        "per_run_metrics": metric_rows,
        "per_run_cluster_bootstrap": bootstrap_rows,
        "across_run_descriptive_differences": across_seeds,
        "outputs": summary["outputs"],
    }, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", nargs="+", required=True,
        help="Path(s) to gdsc2_pair_gnn_ablation_oof_predictions.csv; one per seed/run.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/ai/gdsc2_pair_gnn_uncertainty",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
