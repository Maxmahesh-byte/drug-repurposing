#!/usr/bin/env python3
"""Train and evaluate GDSC2 expression models on held-out cell lines."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
RESPONSE_SOURCE = ROOT / "data/raw/gdsc/gdsc2_responses.csv"
FOLD_SOURCE = ROOT / "data/ai/baselines/gdsc2_ln_ic50_oof_predictions.csv"
OUTPUT_DIR = ROOT / "data/ai/molecular_models"
MODEL_OUTPUT = OUTPUT_DIR / "gdsc2_expression_ridge.json"
SUMMARY_OUTPUT = OUTPUT_DIR / "gdsc2_expression_ridge_summary.json"
PREDICTION_OUTPUTS = {
    "ln_ic50": OUTPUT_DIR / "gdsc2_expression_ln_ic50_oof_predictions.csv",
    "auc": OUTPUT_DIR / "gdsc2_expression_auc_oof_predictions.csv",
}
EXPRESSION_URL = (
    "https://www.ebi.ac.uk/gxa/experiments-content/E-MTAB-2770/resources/"
    "ExperimentDownloadSupplier.RnaSeqBaseline/tpmss.tsv"
)

TARGETS = {"ln_ic50": "ln_ic50", "auc": "auc"}
N_SPLITS = 5
N_FEATURES = 200
RIDGE_ALPHA = 100.0
MIN_TRAIN_MODELS = 20
MIN_DRUG_CANCER_MODELS = 10


def _read_responses() -> pd.DataFrame:
    if not RESPONSE_SOURCE.is_file():
        raise FileNotFoundError(f"GDSC2 response data is missing: {RESPONSE_SOURCE}")
    if not FOLD_SOURCE.is_file():
        raise FileNotFoundError(f"Validated baseline folds are missing: {FOLD_SOURCE}")

    folds: dict[str, int] = {}
    with FOLD_SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            curve_id = row["curve_id"].strip()
            if not curve_id or curve_id in folds:
                raise ValueError(f"Empty or duplicate curve ID in fold file: {curve_id!r}")
            folds[curve_id] = int(row["fold"])

    responses = pd.read_csv(RESPONSE_SOURCE, dtype={"curve_id": str, "drug_id": str})
    required = {
        "curve_id",
        "drug_id",
        "drug_name",
        "cell_line",
        "sanger_model_id",
        "reported_cancer_type",
        *TARGETS.values(),
    }
    absent = required - set(responses.columns)
    if absent:
        raise ValueError(f"GDSC2 response file is missing columns: {sorted(absent)}")
    if responses["curve_id"].duplicated().any():
        raise ValueError("GDSC2 response file contains duplicate curve IDs.")
    if set(responses["curve_id"].astype(str)) != set(folds):
        raise ValueError("GDSC2 source curve IDs differ from the frozen baseline folds.")

    responses["curve_id"] = responses["curve_id"].astype(str)
    responses["fold"] = responses["curve_id"].map(folds)
    for column in ("drug_id", "drug_name", "cell_line", "sanger_model_id"):
        responses[column] = responses[column].fillna("").astype(str).str.strip()
    responses["reported_cancer_type"] = (
        responses["reported_cancer_type"].fillna("").astype(str).str.strip()
    )
    if responses[["drug_id", "cell_line", "sanger_model_id"]].eq("").any().any():
        raise ValueError("GDSC2 response data contain missing drug or model identities.")

    if responses.duplicated(["drug_id", "sanger_model_id"]).any():
        raise ValueError("GDSC2 contains multiple records for a drug/model pair.")
    if responses.groupby("sanger_model_id")["fold"].nunique().gt(1).any():
        raise ValueError("A Sanger model ID is assigned to multiple held-out folds.")

    for target in TARGETS.values():
        responses[target] = pd.to_numeric(responses[target], errors="coerce")
        if not np.isfinite(responses[target].to_numpy(dtype=float)).all():
            raise ValueError(f"GDSC2 target {target!r} contains missing or non-finite values.")
    return responses


def _read_expression_matrix() -> tuple[list[str], list[str], np.ndarray, str]:
    request = requests.get(
        EXPRESSION_URL,
        headers={"User-Agent": "drug-repurposing-research/0.1"},
        stream=True,
        timeout=(15, 120),
    )
    request.raise_for_status()

    sample_labels: list[str] | None = None
    gene_ids: list[str] = []
    gene_rows: list[np.ndarray] = []
    seen_gene_ids: set[str] = set()
    digest = hashlib.sha256()

    try:
        for raw_line in request.iter_lines(chunk_size=1024 * 1024):
            if not raw_line or raw_line.startswith(b"#"):
                continue
            row = next(csv.reader([raw_line.decode("utf-8")], delimiter="\t"))
            if sample_labels is None:
                if row[:2] != ["Gene ID", "Gene Name"]:
                    raise ValueError("Unexpected Expression Atlas matrix header.")
                sample_labels = row[2:]
                if len(sample_labels) != 1019:
                    raise ValueError(
                        f"Expected 1,019 expression samples, found {len(sample_labels)}."
                    )
                digest.update(raw_line)
                digest.update(b"\n")
                continue

            if len(row) != len(sample_labels) + 2:
                raise ValueError("Expression Atlas matrix row has an unexpected width.")
            gene_id = row[0].strip()
            if not gene_id or gene_id in seen_gene_ids:
                raise ValueError(f"Empty or duplicate expression gene ID: {gene_id!r}")
            seen_gene_ids.add(gene_id)
            if len(seen_gene_ids) % 1000 == 0:
                print(
                    f"Streamed {len(seen_gene_ids):,} expression rows...",
                    flush=True,
                )
            try:
                values = np.asarray(
                    [float(value) if value else 0.0 for value in row[2:]],
                    dtype=np.float32,
                )
            except ValueError as error:
                raise ValueError(f"Non-numeric TPM value for gene {gene_id}.") from error
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"Invalid TPM values for gene {gene_id}.")
            digest.update(raw_line)
            digest.update(b"\n")
            log_values = np.log2(values + 1.0).astype(np.float32)
            if np.ptp(log_values) > 0:
                gene_ids.append(gene_id)
                gene_rows.append(log_values)
    finally:
        request.close()

    if sample_labels is None or not gene_rows:
        raise ValueError("Expression Atlas returned no usable expression matrix.")
    return sample_labels, gene_ids, np.stack(gene_rows), digest.hexdigest()


def _build_exact_name_crosswalk(
    responses: pd.DataFrame, sample_labels: list[str]
) -> tuple[dict[str, int], dict[str, int]]:
    expression_columns: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(sample_labels):
        name = label.rpartition(", ")[0].strip()
        if name:
            expression_columns[name.casefold()].append(index)

    response_names: dict[str, set[str]] = defaultdict(set)
    model_names: dict[str, set[str]] = defaultdict(set)
    for row in responses[["sanger_model_id", "cell_line"]].drop_duplicates().itertuples(
        index=False
    ):
        model_id, cell_line = row
        response_names[cell_line.casefold()].add(model_id)
        model_names[model_id].add(cell_line.casefold())

    if any(len(names) != 1 for names in model_names.values()):
        raise ValueError("A Sanger model ID maps to multiple GDSC2 cell-line names.")

    model_to_expression: dict[str, int] = {}
    ambiguous_names = 0
    for name, model_ids in response_names.items():
        columns = expression_columns.get(name, [])
        if len(model_ids) > 1 or len(columns) > 1:
            ambiguous_names += 1
            continue
        if len(columns) == 1:
            model_to_expression[next(iter(model_ids))] = columns[0]
    return model_to_expression, {
        "response_models": len(model_names),
        "matched_models": len(model_to_expression),
        "unmatched_models": len(model_names) - len(model_to_expression),
        "ambiguous_names": ambiguous_names,
        "expression_samples": len(sample_labels),
    }


def _select_fold_features(
    expression: np.ndarray,
    training_columns: list[int],
    n_features: int = N_FEATURES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    variance = np.var(expression[:, training_columns], axis=1, dtype=np.float64)
    variable = np.flatnonzero(variance > 0)
    if variable.size == 0:
        raise ValueError("No variable expression features in the training fold.")
    order = np.lexsort((variable, -variance[variable]))
    selected = variable[order[: min(n_features, len(variable))]]
    values = expression[selected][:, training_columns]
    means = values.mean(axis=1, dtype=np.float64)
    scales = values.std(axis=1, dtype=np.float64)
    keep = scales > 1e-8
    return selected[keep], means[keep], scales[keep]


def _baseline_predictions(
    train: pd.DataFrame, test: pd.DataFrame, target: str
) -> dict[int, dict[str, float]]:
    global_median = float(train[target].median())
    drug_medians = train.groupby("drug_id")[target].median().to_dict()
    qualified: dict[tuple[str, str], float] = {}
    with_cancer = train[train["reported_cancer_type"].ne("")]
    for (drug_id, cancer), values in with_cancer.groupby(
        ["drug_id", "reported_cancer_type"]
    )[target]:
        if len(values) >= MIN_DRUG_CANCER_MODELS:
            qualified[(drug_id, cancer)] = float(values.median())

    predictions: dict[int, dict[str, float]] = {}
    for row in test.itertuples():
        drug_id = row.drug_id
        cancer = row.reported_cancer_type
        drug_median = float(drug_medians.get(drug_id, global_median))
        pair_median = qualified.get((drug_id, cancer), drug_median) if cancer else drug_median
        predictions[row.Index] = {
            "baseline_global_median": global_median,
            "baseline_drug_median": drug_median,
            "baseline_drug_cancer_median": float(pair_median),
        }
    return predictions


def _fit_fold_models(
    matched: pd.DataFrame,
    expression_features: np.ndarray,
    model_to_row: dict[str, int],
    target: str,
    fold: int,
) -> dict[int, float]:
    train = matched[matched["fold"] != fold]
    test = matched[matched["fold"] == fold]
    predictions: dict[int, float] = {}
    for drug_id, test_drug in test.groupby("drug_id", sort=False):
        train_drug = train[train["drug_id"] == drug_id]
        if train_drug["sanger_model_id"].nunique() < MIN_TRAIN_MODELS:
            continue
        training_rows = [
            model_to_row[model_id] for model_id in train_drug["sanger_model_id"]
        ]
        test_rows = [
            model_to_row[model_id] for model_id in test_drug["sanger_model_id"]
        ]
        model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr")
        model.fit(expression_features[training_rows], train_drug[target].to_numpy())
        for row_index, value in zip(test_drug.index, model.predict(expression_features[test_rows])):
            predictions[int(row_index)] = float(value)
    return predictions


def _metrics(frame: pd.DataFrame, prediction_column: str, target: str) -> dict[str, Any]:
    eligible = frame[frame[prediction_column].notna()]
    if eligible.empty:
        raise ValueError(f"No predictions available for {prediction_column}.")
    actual = eligible[target].to_numpy(dtype=float)
    predicted = eligible[prediction_column].to_numpy(dtype=float)
    total = float(np.sum((actual - actual.mean()) ** 2))
    per_drug = eligible.assign(
        absolute_error=np.abs(actual - predicted)
    ).groupby("drug_id")["absolute_error"].mean()
    return {
        "n": int(len(eligible)),
        "n_drugs": int(eligible["drug_id"].nunique()),
        "mae": float(np.mean(np.abs(actual - predicted))),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "r2": None if total == 0 else float(1 - np.sum((actual - predicted) ** 2) / total),
        "macro_drug_mae": float(per_drug.mean()),
    }


def _fit_final_models(
    matched: pd.DataFrame,
    expression: np.ndarray,
    gene_ids: list[str],
    model_to_expression: dict[str, int],
) -> dict[str, Any]:
    all_models = sorted(model_to_expression)
    columns = [model_to_expression[model_id] for model_id in all_models]
    selected, means, scales = _select_fold_features(expression, columns)
    expression_features = (
        (expression[selected][:, columns].T.astype(np.float64) - means) / scales
    )
    model_to_row = {model_id: i for i, model_id in enumerate(all_models)}

    models: dict[str, dict[str, Any]] = {}
    for target in TARGETS.values():
        models[target] = {}
        for drug_id, drug_rows in matched.groupby("drug_id", sort=True):
            if drug_rows["sanger_model_id"].nunique() < MIN_TRAIN_MODELS:
                continue
            row_indices = [
                model_to_row[model_id] for model_id in drug_rows["sanger_model_id"]
            ]
            model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr")
            model.fit(expression_features[row_indices], drug_rows[target].to_numpy())
            models[target][drug_id] = {
                "drug_name": str(drug_rows["drug_name"].iloc[0]),
                "training_models": int(drug_rows["sanger_model_id"].nunique()),
                "feature_ids": [gene_ids[index] for index in selected],
                "feature_means": means.tolist(),
                "feature_scales": scales.tolist(),
                "coefficients": model.coef_.tolist(),
                "intercept": float(model.intercept_),
            }
    return models


def main() -> None:
    responses = _read_responses()
    sample_labels, gene_ids, expression, expression_digest = _read_expression_matrix()
    model_to_expression, crosswalk = _build_exact_name_crosswalk(
        responses, sample_labels
    )
    matched = responses[
        responses["sanger_model_id"].isin(model_to_expression)
    ].copy().reset_index(drop=True)
    if matched.empty:
        raise ValueError("No GDSC2 cell lines matched Expression Atlas by exact name.")

    fold_counts = matched.groupby("sanger_model_id")["fold"].nunique()
    if fold_counts.gt(1).any():
        raise ValueError("Matched cell lines occur in multiple held-out folds.")
    fold_by_model = matched.groupby("sanger_model_id")["fold"].first().to_dict()
    expression_columns = [
        model_to_expression[model_id] for model_id in sorted(model_to_expression)
    ]
    model_order = sorted(model_to_expression)
    model_to_row = {model_id: index for index, model_id in enumerate(model_order)}
    output_frames: dict[str, pd.DataFrame] = {}
    fold_feature_ids: dict[str, list[str]] = {}
    fold_reports: list[dict[str, Any]] = []

    for fold in range(1, N_SPLITS + 1):
        train_models = sorted(
            model_id for model_id in model_order if int(fold_by_model[model_id]) != fold
        )
        training_columns = [model_to_expression[model_id] for model_id in train_models]
        selected, means, scales = _select_fold_features(expression, training_columns)
        fold_feature_ids[str(fold)] = [gene_ids[index] for index in selected]
        fold_expression = (
            (expression[selected][:, expression_columns].T.astype(np.float64) - means)
            / scales
        )
        fold_train = matched[matched["fold"] != fold]
        fold_test = matched[matched["fold"] == fold]
        baseline_by_target = {
            target: _baseline_predictions(fold_train, fold_test, target)
            for target in TARGETS.values()
        }
        fold_reports.append(
            {
                "fold": fold,
                "training_models": len(train_models),
                "test_models": int(fold_test["sanger_model_id"].nunique()),
                "test_rows": int(len(fold_test)),
            }
        )

        for endpoint, target in TARGETS.items():
            if endpoint not in output_frames:
                output_frames[endpoint] = matched[
                    [
                        "curve_id",
                        "drug_id",
                        "drug_name",
                        "cell_line",
                        "sanger_model_id",
                        "reported_cancer_type",
                        "fold",
                        target,
                    ]
                ].copy()
                for column in (
                    "baseline_global_median",
                    "baseline_drug_median",
                    "baseline_drug_cancer_median",
                    "expression_ridge",
                ):
                    output_frames[endpoint][column] = np.nan

            predictions = _fit_fold_models(
                matched,
                fold_expression,
                model_to_row,
                target,
                fold,
            )
            frame = output_frames[endpoint]
            for row_index, baseline in baseline_by_target[target].items():
                for name, value in baseline.items():
                    frame.loc[row_index, name] = value
            for row_index, value in predictions.items():
                frame.loc[row_index, "expression_ridge"] = value

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    endpoint_metrics: dict[str, Any] = {}
    for endpoint, frame in output_frames.items():
        model_metrics = _metrics(frame, "expression_ridge", TARGETS[endpoint])
        paired = frame[frame["expression_ridge"].notna()]
        baseline_metrics = {
            column: _metrics(paired, column, TARGETS[endpoint])
            for column in (
                "baseline_global_median",
                "baseline_drug_median",
                "baseline_drug_cancer_median",
            )
        }
        endpoint_metrics[endpoint] = {
            "expression_ridge": model_metrics,
            "baselines_on_same_rows": baseline_metrics,
            "macro_drug_mae_difference_vs_drug_cancer_baseline": (
                model_metrics["macro_drug_mae"]
                - baseline_metrics["baseline_drug_cancer_median"]["macro_drug_mae"]
            ),
        }
        frame.to_csv(PREDICTION_OUTPUTS[endpoint], index=False)

    final_models = _fit_final_models(
        matched, expression, gene_ids, model_to_expression
    )
    source_digest = hashlib.sha256(RESPONSE_SOURCE.read_bytes()).hexdigest()
    bundle = {
        "schema_version": 1,
        "model_type": "per-drug ridge regression on CCLE RNA-seq expression",
        "expression_source": "Expression Atlas E-MTAB-2770",
        "expression_url": EXPRESSION_URL,
        "expression_gene_rows_sha256": expression_digest,
        "response_source": str(RESPONSE_SOURCE.relative_to(ROOT)),
        "response_source_sha256": source_digest,
        "matching": "case-insensitive exact cell-line name; ambiguous names excluded",
        "training_scope": "known GDSC2 drugs; new cell lines represented by RNA-seq TPM",
        "feature_transform": "log2(TPM + 1), then training-cohort z-score",
        "feature_selection": f"top {N_FEATURES} variable genes in training cohort",
        "ridge_alpha": RIDGE_ALPHA,
        "minimum_training_models_per_drug": MIN_TRAIN_MODELS,
        "targets": final_models,
    }
    MODEL_OUTPUT.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    summary = {
        "status": "evaluated",
        "crosswalk": crosswalk,
        "matched_response_rows": int(len(matched)),
        "matched_drugs": int(matched["drug_id"].nunique()),
        "fold_method": "existing five-fold GroupKFold by Sanger model ID",
        "folds": fold_reports,
        "fold_feature_ids": fold_feature_ids,
        "model": {
            "type": "separate Ridge model per drug and response endpoint",
            "feature_count": N_FEATURES,
            "alpha": RIDGE_ALPHA,
            "minimum_training_models_per_drug": MIN_TRAIN_MODELS,
            "feature_selection_is_training_fold_only": True,
            "response_target_used_as_feature": False,
        },
        "metrics": endpoint_metrics,
        "outputs": {
            "model_bundle": str(MODEL_OUTPUT.relative_to(ROOT)),
            "ln_ic50_predictions": str(PREDICTION_OUTPUTS["ln_ic50"].relative_to(ROOT)),
            "auc_predictions": str(PREDICTION_OUTPUTS["auc"].relative_to(ROOT)),
        },
        "interpretation": (
            "This is an exploratory benchmark for known drugs on an exact-name-matched "
            "subset of cell lines. It does not validate unseen-drug generalisation, "
            "patient response, clinical efficacy, or drug combinations. Molecular-model "
            "utility requires improvement over the response-only baselines on held-out "
            "cell lines and independent validation."
        ),
    }
    SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
