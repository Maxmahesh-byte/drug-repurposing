#!/usr/bin/env python3
"""Run fold-matched ablations for the GDSC2 drug/cell-line response model.

Variants:
  full_graph: RGCN drug encoder + expression encoder + pair head.
  drug_id_only: learned drug-ID embeddings + expression encoder, no graph edges.
  expression_only: expression encoder only, no drug identity or graph.

All variants use the same outer and inner cell-line splits, fold-specific
expression preprocessing, and training-fold-only endpoint standardisation.
Metrics are reported after inverse-transforming predictions to original units.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ai.train_gdsc2_pair_gnn import (
    DEFAULT_DATA,
    DEFAULT_GRAPH,
    metric_block,
    select_and_scale_expression,
    set_seed,
    sha256_file,
)
from src.training.gdsc2_pair_gnn import DrugCellLinePairGNN, build_compact_drug_graph
from src.training.gdsc2_pair_ablations import DrugIdExpressionModel, ExpressionOnlyModel

ENDPOINTS = ("ln_ic50", "auc")
VARIANTS = ("full_graph", "drug_id_only", "expression_only")


def fit_target_scaler(training_targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit two independent target scalers using only the inner-fit records."""
    values = np.asarray(training_targets, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Training targets must have shape (rows, 2).")
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Training targets must be non-empty and finite.")
    means = values.mean(axis=0)
    scales = values.std(axis=0, ddof=0)
    scales[scales < 1e-8] = 1.0
    return means.astype(np.float32), scales.astype(np.float32)


def validate_pair_expression_alignment(
    pairs: pd.DataFrame,
    model_ids: np.ndarray,
) -> None:
    """Fail closed if pair rows do not point to their matching expression rows."""
    required = {
        "curve_id", "sanger_model_id", "graph_node_id",
        "expression_row_index", "fold", "drug_id", "ln_ic50", "auc",
    }
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"Response-pair table is missing columns: {sorted(missing)}")
    canonical_ids = [str(value).strip() for value in model_ids.tolist()]
    if any(not value for value in canonical_ids):
        raise ValueError("Expression metadata contains an empty Sanger model ID.")
    if len(canonical_ids) != len(set(canonical_ids)):
        raise ValueError("Expression metadata contains duplicate Sanger model IDs.")
    raw_pair_ids = pairs["sanger_model_id"].astype(str)
    if not raw_pair_ids.equals(raw_pair_ids.str.strip()):
        raise ValueError("Response-pair Sanger model IDs contain surrounding whitespace.")
    pair_ids = raw_pair_ids
    raw_drug_ids = pairs["drug_id"].astype(str)
    if raw_drug_ids.str.strip().eq("").any() or not raw_drug_ids.equals(raw_drug_ids.str.strip()):
        raise ValueError("Response-pair drug IDs are blank or contain surrounding whitespace.")
    if pairs["curve_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Response-pair table contains blank curve IDs.")
    if pairs["curve_id"].astype(str).duplicated().any():
        raise ValueError("Response-pair table contains duplicate curve IDs.")
    row_lookup = {model_id: index for index, model_id in enumerate(canonical_ids)}
    missing_models = sorted(set(pair_ids) - set(row_lookup))
    if missing_models:
        raise ValueError(
            f"Response Sanger model IDs are absent from expression metadata: {missing_models[:10]}"
        )
    expected_rows = pair_ids.map(row_lookup).to_numpy(dtype=np.int64)
    observed_rows = pd.to_numeric(
        pairs["expression_row_index"], errors="coerce"
    ).to_numpy(dtype=float)
    if (
        not np.isfinite(observed_rows).all()
        or not np.equal(observed_rows, np.floor(observed_rows)).all()
        or not np.array_equal(observed_rows.astype(np.int64), expected_rows)
    ):
        raise ValueError(
            "expression_row_index does not match the NPZ row for its Sanger model ID."
        )
    raw_graph_ids = pairs["graph_node_id"].astype(str)
    if not raw_graph_ids.equals(raw_graph_ids.str.strip()):
        raise ValueError("Response-pair graph node IDs contain surrounding whitespace.")
    expected_graph_ids = pair_ids.map(lambda value: f"GDSC_MODEL:{value}")
    if not raw_graph_ids.equals(expected_graph_ids):
        raise ValueError("graph_node_id does not match its Sanger model ID.")
    folds = pd.to_numeric(pairs["fold"], errors="coerce")
    if folds.isna().any() or not folds.isin(range(1, 6)).all():
        raise ValueError("Response-pair rows must have frozen fold IDs 1–5.")
    if pairs.assign(_fold=folds).groupby("sanger_model_id")["_fold"].nunique().gt(1).any():
        raise ValueError("A Sanger model ID is assigned to multiple outer folds.")
    targets = pairs[["ln_ic50", "auc"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(targets).all():
        raise ValueError("Response-pair targets contain missing or non-finite values.")


def split_outer_fold_rows(
    pairs: pd.DataFrame,
    fold: int,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition cell-line groups into inner-fit, inner-validation, and outer-test."""
    if fold not in range(1, 6):
        raise ValueError("Outer fold must be one of the frozen IDs 1–5.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between zero and one.")
    if pairs.groupby("sanger_model_id")["fold"].nunique().gt(1).any():
        raise ValueError("A Sanger model ID is assigned to multiple outer folds.")
    outer_train = pairs[pairs["fold"] != fold].copy()
    outer_test = pairs[pairs["fold"] == fold].copy()
    if outer_train.empty or outer_test.empty:
        raise ValueError(f"Fold {fold} has empty outer train or test rows.")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=validation_fraction, random_state=seed
    )
    fit_index, val_index = next(
        splitter.split(outer_train, groups=outer_train["sanger_model_id"])
    )
    fit_rows = outer_train.iloc[fit_index].copy()
    val_rows = outer_train.iloc[val_index].copy()
    fit_groups = set(fit_rows["sanger_model_id"].astype(str))
    val_groups = set(val_rows["sanger_model_id"].astype(str))
    test_groups = set(outer_test["sanger_model_id"].astype(str))
    if fit_groups & val_groups:
        raise RuntimeError("Inner fit and validation groups overlap.")
    if fit_groups & test_groups:
        raise RuntimeError("Outer test cell lines leaked into inner fit.")
    if val_groups & test_groups:
        raise RuntimeError("Outer test cell lines leaked into inner validation.")
    all_groups = set(pairs["sanger_model_id"].astype(str))
    if fit_groups | val_groups | test_groups != all_groups:
        raise RuntimeError("Fit/validation/test partition does not cover all cell lines.")
    return fit_rows, val_rows, outer_test


def build_model(
    variant: str,
    drug_graph,
    num_drugs: int,
    num_expression_features: int,
    args,
) -> torch.nn.Module:
    if variant == "full_graph":
        return DrugCellLinePairGNN(
            num_graph_nodes=drug_graph.num_nodes,
            num_node_types=len(drug_graph.node_counts),
            num_relations=drug_graph.num_relations,
            num_drugs=num_drugs,
            num_expression_features=num_expression_features,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
    if variant == "drug_id_only":
        return DrugIdExpressionModel(
            num_drugs=num_drugs,
            num_expression_features=num_expression_features,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
    if variant == "expression_only":
        return ExpressionOnlyModel(
            num_expression_features=num_expression_features,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
    raise ValueError(f"Unknown ablation variant: {variant}")


def validate_full_outer_refit_partition(
    fit_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    outer_train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
) -> set[str]:
    """Require final refit rows to contain every outer-train group and no test group."""
    fit_groups = set(fit_rows["sanger_model_id"].astype(str))
    val_groups = set(val_rows["sanger_model_id"].astype(str))
    train_groups = set(outer_train_rows["sanger_model_id"].astype(str))
    test_groups = set(test_rows["sanger_model_id"].astype(str))
    if fit_groups & val_groups:
        raise RuntimeError("Inner fit and validation groups overlap.")
    if train_groups != (fit_groups | val_groups):
        raise RuntimeError(
            "Complete outer-training groups are not exactly the union of inner fit and validation."
        )
    if train_groups & test_groups:
        raise RuntimeError("Outer test cell lines leaked into final refit.")
    if not train_groups or not test_groups:
        raise ValueError("Final refit requires non-empty outer-training and outer-test groups.")
    return train_groups


def prepare_full_outer_refit(
    raw_expression: np.ndarray,
    model_ids: np.ndarray,
    outer_train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    top_k_genes: int,
) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray]:
    """Fit expression and target preprocessing on the complete outer-training fold."""
    if outer_train_rows.empty:
        raise ValueError("Cannot refit on an empty outer-training fold.")
    train_groups = set(outer_train_rows["sanger_model_id"].astype(str))
    test_groups = set(test_rows["sanger_model_id"].astype(str))
    if train_groups & test_groups:
        raise ValueError("Outer test cell lines cannot be used for final-refit preprocessing.")
    refit_expression, feature_meta = select_and_scale_expression(
        raw_expression, model_ids, train_groups, top_k_genes
    )
    target_values = outer_train_rows[list(ENDPOINTS)].to_numpy(dtype=np.float32)
    target_mean, target_scale = fit_target_scaler(target_values)
    feature_meta["fit_scope"] = "complete outer-training cell lines"
    feature_meta["training_cell_lines_for_scaler"] = int(len(train_groups))
    return refit_expression, feature_meta, target_mean, target_scale


def run_variant(
    variant: str,
    fold: int,
    fit_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    outer_train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    raw_expression: np.ndarray,
    scaled_expression: np.ndarray,
    model_ids: np.ndarray,
    drug_graph,
    drug_id_to_index: dict[str, int],
    feature_meta: dict,
    args,
    device: torch.device,
) -> tuple[pd.DataFrame, dict, dict]:
    """Select epoch on inner split, then refit from scratch on all outer-train rows."""
    selection_seed = args.seed + fold * 101
    set_seed(selection_seed)
    num_drugs = len(drug_graph.drug_ids)
    model = build_model(
        variant, drug_graph, num_drugs, scaled_expression.shape[1], args
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    model_row_lookup = {
        str(model_id): index for index, model_id in enumerate(model_ids.astype(str))
    }
    edge_index = drug_graph.edge_index.to(device)
    edge_type = drug_graph.edge_type.to(device)
    node_type = drug_graph.node_type.to(device)

    def pair_indices_for(frame: pd.DataFrame):
        drugs = torch.tensor(
            [drug_id_to_index[str(value)] for value in frame["drug_id"]],
            dtype=torch.long,
            device=device,
        )
        cells = torch.tensor(
            [model_row_lookup[str(value)] for value in frame["sanger_model_id"]],
            dtype=torch.long,
            device=device,
        )
        return drugs, cells

    fit_drugs, fit_cells = pair_indices_for(fit_rows)
    val_drugs, val_cells = pair_indices_for(val_rows)
    fit_targets_raw = torch.tensor(
        fit_rows[list(ENDPOINTS)].to_numpy(dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    val_targets_raw = torch.tensor(
        val_rows[list(ENDPOINTS)].to_numpy(dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )

    selection_target_mean, selection_target_scale = fit_target_scaler(
        fit_targets_raw.detach().cpu().numpy()
    )
    selection_mean_tensor = torch.from_numpy(selection_target_mean).to(device)
    selection_scale_tensor = torch.from_numpy(selection_target_scale).to(device)
    fit_targets = (fit_targets_raw - selection_mean_tensor) / selection_scale_tensor
    val_targets = (val_targets_raw - selection_mean_tensor) / selection_scale_tensor
    selection_expression_tensor = torch.from_numpy(scaled_expression).to(device)

    def predict_selection(drug_indices, cell_indices):
        model.eval()
        if variant == "full_graph":
            drug_embeddings, cell_embeddings = model.encode(
                selection_expression_tensor, edge_index, edge_type, node_type
            )
            return model.score_pairs(
                drug_embeddings, cell_embeddings, drug_indices, cell_indices
            )
        return model(
            selection_expression_tensor, drug_indices, cell_indices
        )

    best_val = float("inf")
    best_epoch = 0
    patience_left = args.patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if variant == "full_graph":
            drug_embeddings, cell_embeddings = model.encode(
                selection_expression_tensor, edge_index, edge_type, node_type
            )
            fit_predictions = model.score_pairs(
                drug_embeddings, cell_embeddings, fit_drugs, fit_cells
            )
        else:
            fit_predictions = model(
                selection_expression_tensor, fit_drugs, fit_cells
            )
        selection_loss = torch.nn.functional.smooth_l1_loss(
            fit_predictions, fit_targets
        )
        if not torch.isfinite(selection_loss):
            raise RuntimeError(
                f"Non-finite {variant} epoch-selection loss in fold {fold}, epoch {epoch}."
            )
        selection_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        with torch.no_grad():
            validation_predictions = predict_selection(val_drugs, val_cells)
            validation_loss = torch.nn.functional.smooth_l1_loss(
                validation_predictions, val_targets
            ).item()
        if validation_loss < best_val - args.min_delta:
            best_val = validation_loss
            best_epoch = epoch
            patience_left = args.patience
        else:
            patience_left -= 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"fold={fold} variant={variant} selection_epoch={epoch} "
                f"fit_scaled_smooth_l1={selection_loss.item():.5f} "
                f"val_scaled_smooth_l1={validation_loss:.5f}",
                flush=True,
            )
        if patience_left <= 0:
            break

    if best_epoch < 1:
        raise RuntimeError(
            f"No epoch selected for {variant}, fold {fold}; inner validation failed."
        )

    # Refit the selected architecture from a fresh initialisation on every
    # response row and cell line in the complete outer-training fold.
    outer_train_groups = validate_full_outer_refit_partition(
        fit_rows, val_rows, outer_train_rows, test_rows
    )
    (
        refit_expression,
        refit_feature_meta,
        refit_target_mean,
        refit_target_scale,
    ) = prepare_full_outer_refit(
        raw_expression,
        model_ids,
        outer_train_rows,
        test_rows,
        args.top_k_genes,
    )
    if refit_feature_meta["training_cell_lines_for_scaler"] != len(outer_train_groups):
        raise RuntimeError("Final-refit expression scaler did not use all outer-training cell lines.")

    # Use a deterministic but separate initialisation stream for the refit.
    set_seed(selection_seed + 1_000_003)
    refit_model = build_model(
        variant,
        drug_graph,
        num_drugs,
        refit_expression.shape[1],
        args,
    ).to(device)
    refit_optimizer = torch.optim.AdamW(
        refit_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    refit_expression_tensor = torch.from_numpy(refit_expression).to(device)
    refit_train_drugs, refit_train_cells = pair_indices_for(outer_train_rows)
    test_drugs, test_cells = pair_indices_for(test_rows)
    refit_targets_raw = torch.tensor(
        outer_train_rows[list(ENDPOINTS)].to_numpy(dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    refit_mean_tensor = torch.from_numpy(refit_target_mean).to(device)
    refit_scale_tensor = torch.from_numpy(refit_target_scale).to(device)
    refit_targets = (
        refit_targets_raw - refit_mean_tensor
    ) / refit_scale_tensor

    def predict_refit(drug_indices, cell_indices):
        refit_model.eval()
        if variant == "full_graph":
            drug_embeddings, cell_embeddings = refit_model.encode(
                refit_expression_tensor, edge_index, edge_type, node_type
            )
            return refit_model.score_pairs(
                drug_embeddings, cell_embeddings, drug_indices, cell_indices
            )
        return refit_model(
            refit_expression_tensor, drug_indices, cell_indices
        )

    final_refit_loss = float("nan")
    for refit_epoch in range(1, best_epoch + 1):
        refit_model.train()
        refit_optimizer.zero_grad(set_to_none=True)
        if variant == "full_graph":
            drug_embeddings, cell_embeddings = refit_model.encode(
                refit_expression_tensor, edge_index, edge_type, node_type
            )
            refit_predictions = refit_model.score_pairs(
                drug_embeddings,
                cell_embeddings,
                refit_train_drugs,
                refit_train_cells,
            )
        else:
            refit_predictions = refit_model(
                refit_expression_tensor,
                refit_train_drugs,
                refit_train_cells,
            )
        refit_loss = torch.nn.functional.smooth_l1_loss(
            refit_predictions, refit_targets
        )
        if not torch.isfinite(refit_loss):
            raise RuntimeError(
                f"Non-finite {variant} full-fold refit loss in fold {fold}, "
                f"epoch {refit_epoch}."
            )
        refit_loss.backward()
        torch.nn.utils.clip_grad_norm_(refit_model.parameters(), max_norm=2.0)
        refit_optimizer.step()
        final_refit_loss = float(refit_loss.item())
        if refit_epoch == 1 or refit_epoch % 10 == 0 or refit_epoch == best_epoch:
            print(
                f"fold={fold} variant={variant} refit_epoch={refit_epoch}/"
                f"{best_epoch} full_outer_train_scaled_smooth_l1={final_refit_loss:.5f}",
                flush=True,
            )

    refit_model.eval()
    with torch.no_grad():
        scaled_predictions = predict_refit(test_drugs, test_cells).cpu().numpy()
    predictions = (
        scaled_predictions * refit_target_scale[None, :]
        + refit_target_mean[None, :]
    )

    result = test_rows[
        ["curve_id", "drug_id", "sanger_model_id", "fold", "ln_ic50", "auc"]
    ].copy()
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        result[f"pred_{variant}_{endpoint}"] = predictions[:, endpoint_index]

    metrics = {
        "variant": variant,
        "fold": fold,
        "inner_fit_cell_lines": int(fit_rows["sanger_model_id"].nunique()),
        "inner_validation_cell_lines": int(val_rows["sanger_model_id"].nunique()),
        "outer_training_cell_lines": int(outer_train_rows["sanger_model_id"].nunique()),
        "outer_test_cell_lines": int(test_rows["sanger_model_id"].nunique()),
        "inner_fit_response_rows": int(len(fit_rows)),
        "inner_validation_rows": int(len(val_rows)),
        "outer_training_response_rows": int(len(outer_train_rows)),
        "outer_test_rows": int(len(test_rows)),
        "selected_epoch_from_inner_validation": int(best_epoch),
        "best_inner_validation_scaled_smooth_l1": float(best_val),
        "final_refit_epochs": int(best_epoch),
        "final_refit_training_loss": final_refit_loss,
        "final_refit_used_all_outer_training_rows": True,
        "expression_preprocessing": refit_feature_meta,
        "target_scaler": {
            endpoint: {
                "training_mean": float(refit_target_mean[index]),
                "training_std": float(refit_target_scale[index]),
                "fit_scope": "complete outer-training response rows",
            }
            for index, endpoint in enumerate(ENDPOINTS)
        },
        "ln_ic50": metric_block(
            test_rows["ln_ic50"].to_numpy(dtype=float), predictions[:, 0]
        ),
        "auc": metric_block(
            test_rows["auc"].to_numpy(dtype=float), predictions[:, 1]
        ),
    }
    checkpoint = {
        "state_dict": refit_model.state_dict(),
        "variant": variant,
        "fold": fold,
        "message_passing_layers": (
            refit_model.num_message_passing_layers if variant == "full_graph" else 0
        ),
        "selected_epoch_from_inner_validation": int(best_epoch),
        "final_refit_epochs": int(best_epoch),
        "training_scope": "all outer-training Sanger model IDs and response rows",
        "target_mean": refit_target_mean,
        "target_scale": refit_target_scale,
        "model_config": {
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "top_k_genes": args.top_k_genes,
        },
        "selected_gene_indices": refit_feature_meta["selected_gene_indices"],
        "expression_scaler_mean": refit_feature_meta["scaler_mean"],
        "expression_scaler_scale": refit_feature_meta["scaler_scale"],
        "expression_scaler_fit_scope": refit_feature_meta["fit_scope"],
        "selection_stage": {
            "inner_fit_cell_lines": int(fit_rows["sanger_model_id"].nunique()),
            "inner_validation_cell_lines": int(val_rows["sanger_model_id"].nunique()),
            "best_validation_loss": float(best_val),
            "selected_epoch": int(best_epoch),
            "target_mean": selection_target_mean,
            "target_scale": selection_target_scale,
            "expression_feature_indices": feature_meta["selected_gene_indices"],
            "expression_scaler_mean": feature_meta["scaler_mean"],
            "expression_scaler_scale": feature_meta["scaler_scale"],
        },
    }
    return result, metrics, checkpoint

def compare_external_baselines(
    oof: pd.DataFrame,
    baseline_specs: dict[str, tuple[Path, str, str]],
) -> dict:
    comparisons = {}
    for name, (path, prediction_column, actual_column) in baseline_specs.items():
        if not path.is_file():
            comparisons[name] = {"status": "missing", "path": str(path)}
            continue
        baseline = pd.read_csv(
            path,
            dtype={"curve_id": str, "drug_id": str, "sanger_model_id": str},
        )
        required = {
            "curve_id", "fold", "sanger_model_id", prediction_column, actual_column
        }
        if required - set(baseline.columns):
            comparisons[name] = {
                "status": "missing_required_columns",
                "path": str(path),
                "missing_columns": sorted(required - set(baseline.columns)),
                "available_columns": list(baseline.columns),
            }
            continue
        if baseline["curve_id"].duplicated().any():
            raise ValueError(f"Duplicate curve IDs in comparator {path}.")
        endpoint = "ln_ic50" if "ln_ic50" in name else "auc"
        variant_cols = [f"pred_{variant}_{endpoint}" for variant in VARIANTS]
        selected_baseline = baseline[
            ["curve_id", "fold", "sanger_model_id", actual_column, prediction_column]
        ].rename(
            columns={
                "fold": "fold_baseline",
                "sanger_model_id": "sanger_model_id_baseline",
                actual_column: "comparator_actual",
            }
        )
        joined = oof[
            ["curve_id", "fold", "sanger_model_id", f"actual_{endpoint}", *variant_cols]
        ].merge(
            selected_baseline,
            on="curve_id",
            how="inner",
            validate="one_to_one",
            suffixes=("_oof", "_baseline"),
        )
        comparison_key = f"{name}_{endpoint}"
        if len(joined) == 0:
            comparisons[comparison_key] = {"status": "no_overlap"}
            continue
        fold_match = np.array_equal(
            joined["fold"].to_numpy(), joined["fold_baseline"].to_numpy()
        )
        if not fold_match:
            raise ValueError(f"Fold assignments differ in comparator {name}.")
        model_id_match = joined["sanger_model_id"].astype(str).equals(
            joined["sanger_model_id_baseline"].astype(str)
        )
        if not model_id_match:
            raise ValueError(f"Sanger model IDs differ in comparator {name}.")
        # Ensure both files agree on the measured target before scoring.
        actual_oof = joined[f"actual_{endpoint}"].to_numpy(dtype=float)
        actual_comparator = joined["comparator_actual"].to_numpy(dtype=float)
        if not np.allclose(actual_oof, actual_comparator, rtol=0, atol=1e-8):
            raise ValueError(f"Measured target values differ in comparator {name}.")
        prediction_metrics = {
            variant: metric_block(
                actual_oof,
                joined[f"pred_{variant}_{endpoint}"].to_numpy(dtype=float),
            )
            for variant in VARIANTS
        }
        prediction_metrics["external_baseline"] = metric_block(
            actual_oof, joined[prediction_column].to_numpy(dtype=float)
        )
        comparisons[comparison_key] = {
            "status": "compared",
            "overlap_rows": int(len(joined)),
            "total_oof_rows": int(len(oof)),
            "coverage_fraction": float(len(joined) / len(oof)),
            "fold_assignments_identical": fold_match,
            "sanger_model_ids_identical": model_id_match,
            "measured_targets_identical": True,
            "metrics": prediction_metrics,
        }
    return comparisons


def main(args) -> dict:
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    pair_path = data_dir / "gdsc2_matched_response_pairs.csv"
    expression_path = data_dir / "gdsc2_matched_expression_features.npz"
    graph_path = Path(args.graph_file)
    for path in (pair_path, expression_path, graph_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    pairs = pd.read_csv(
        pair_path,
        dtype={"curve_id": str, "drug_id": str, "sanger_model_id": str},
    )
    with np.load(expression_path, allow_pickle=False) as archive:
        expression = archive["expression_log2_tpm"].astype(np.float32)
        gene_ids = archive["gene_ids"].astype(str)
        model_ids = archive["sanger_model_ids"].astype(str)
    if expression.shape != (len(model_ids), len(gene_ids)):
        raise ValueError("Expression matrix dimensions do not align with metadata.")
    if not np.isfinite(expression).all():
        raise ValueError("Expression matrix contains non-finite values.")
    canonical_gene_ids = [str(value).strip() for value in gene_ids.tolist()]
    if any(not value for value in canonical_gene_ids):
        raise ValueError("Expression gene IDs contain blank identifiers.")
    if len(canonical_gene_ids) != len(set(canonical_gene_ids)):
        raise ValueError("Expression gene IDs contain duplicates.")
    if pairs["curve_id"].duplicated().any():
        raise ValueError("Response-pair table contains duplicate curve IDs.")
    if set(pairs["fold"].unique()) != set(range(1, 6)):
        raise ValueError("Expected frozen fold IDs 1–5.")
    if set(pairs["sanger_model_id"].astype(str)) - set(model_ids.astype(str)):
        raise ValueError("Some response Sanger model IDs are absent from expression matrix metadata.")

    if len(gene_ids) != len(set(gene_ids.astype(str).tolist())):
        raise ValueError("Expression gene IDs contain duplicates.")
    validate_pair_expression_alignment(pairs, model_ids)

    print("Loading typed graph and extracting response-free drug evidence...", flush=True)
    typed_graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    drug_graph = build_compact_drug_graph(typed_graph)
    from src.training.gdsc2_pair_gnn import summarize_drug_evidence_coverage
    evidence_coverage = summarize_drug_evidence_coverage(drug_graph)
    drug_id_to_index = {
        node_id.split(":", 1)[1]: index
        for index, node_id in enumerate(drug_graph.drug_ids)
        if node_id.startswith("GDSC_DRUG:")
    }
    if len(drug_id_to_index) != len(drug_graph.drug_ids):
        raise ValueError("Unexpected drug node ID format in typed graph.")
    missing_drugs = sorted(set(pairs["drug_id"].astype(str)) - set(drug_id_to_index))
    if missing_drugs:
        raise ValueError(f"Response drugs missing graph nodes: {missing_drugs[:10]}")
    graph_cell_ids = set(str(value) for value in typed_graph["cell_line"].node_ids)
    missing_cells = sorted(set(pairs["graph_node_id"].astype(str)) - graph_cell_ids)
    if missing_cells:
        raise ValueError(f"Matched response cell-line graph nodes missing: {missing_cells[:10]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device={device}; variants={', '.join(VARIANTS)}; "
        f"response_pairs={len(pairs)}",
        flush=True,
    )

    variant_oof: dict[str, list[pd.DataFrame]] = {variant: [] for variant in VARIANTS}
    fold_metrics: list[dict] = []
    for fold in range(1, 6):
        fit_rows, val_rows, outer_test = split_outer_fold_rows(
            pairs,
            fold,
            args.validation_fraction,
            args.seed + fold,
        )
        fit_groups = set(fit_rows["sanger_model_id"].astype(str))
        scaled_expression, feature_meta = select_and_scale_expression(
            expression, model_ids, fit_groups, args.top_k_genes
        )
        for variant in VARIANTS:
            print(f"\n===== Fold {fold}/5 — {variant} =====", flush=True)
            predictions, metrics, checkpoint = run_variant(
                variant,
                fold,
                fit_rows,
                val_rows,
                pairs[pairs["fold"] != fold].copy(),
                outer_test,
                expression,
                scaled_expression,
                model_ids,
                drug_graph,
                drug_id_to_index,
                feature_meta,
                args,
                device,
            )
            variant_oof[variant].append(predictions)
            metrics["inner_selection_expression_preprocessing"] = feature_meta
            fold_metrics.append(metrics)
            torch.save(
                checkpoint,
                output_dir / f"{variant}_fold_{fold}_best_state.pt",
            )
            print(
                f"fold={fold} variant={variant} "
                f"LN_IC50_MAE={metrics['ln_ic50']['mae']:.5f} "
                f"AUC_MAE={metrics['auc']['mae']:.5f}",
                flush=True,
            )

    combined = None
    for variant, frames in variant_oof.items():
        variant_frame = pd.concat(frames, ignore_index=True)
        if variant_frame["curve_id"].duplicated().any():
            raise ValueError(f"Duplicate OOF curve IDs for {variant}.")
        renamed = variant_frame.rename(
            columns={
                "ln_ic50": "actual_ln_ic50",
                "auc": "actual_auc",
            }
        )
        prediction_columns = [
            f"pred_{variant}_{endpoint}" for endpoint in ENDPOINTS
        ]
        keep = ["curve_id", "drug_id", "sanger_model_id", "fold", "actual_ln_ic50", "actual_auc", *prediction_columns]
        renamed = renamed[keep]
        combined = renamed if combined is None else combined.merge(
            renamed.drop(columns=["drug_id", "sanger_model_id", "fold", "actual_ln_ic50", "actual_auc"]),
            on="curve_id",
            how="inner",
            validate="one_to_one",
        )

    expected_rows = len(pairs)
    if combined is None or len(combined) != expected_rows:
        raise ValueError(
            f"Ablation OOF predictions cover {0 if combined is None else len(combined)} "
            f"curves; expected {expected_rows}."
        )
    if combined["curve_id"].duplicated().any():
        raise ValueError("Combined ablation OOF file contains duplicate curves.")
    prediction_path = output_dir / "gdsc2_pair_gnn_ablation_oof_predictions.csv"
    metrics_path = output_dir / "gdsc2_pair_gnn_ablation_metrics.json"
    combined.to_csv(prediction_path, index=False)

    overall = {}
    for endpoint in ENDPOINTS:
        actual = combined[f"actual_{endpoint}"].to_numpy(dtype=float)
        overall[endpoint] = {
            variant: metric_block(
                actual,
                combined[f"pred_{variant}_{endpoint}"].to_numpy(dtype=float),
            )
            for variant in VARIANTS
        }

    baseline_dir = ROOT / "data/ai/baselines"
    molecular_dir = ROOT / "data/ai/molecular_models"
    baseline_comparison = compare_external_baselines(
        combined,
        {
            "drug_cancer_median_ln_ic50": (
                baseline_dir / "gdsc2_ln_ic50_oof_predictions.csv",
                "B2_drug_cancer_median",
                "actual_ln_ic50",
            ),
            "drug_cancer_median_auc": (
                baseline_dir / "gdsc2_auc_oof_predictions.csv",
                "B2_drug_cancer_median",
                "actual_auc",
            ),
            "expression_ridge_ln_ic50": (
                molecular_dir / "gdsc2_expression_ln_ic50_oof_predictions.csv",
                "expression_ridge",
                "ln_ic50",
            ),
            "expression_ridge_auc": (
                molecular_dir / "gdsc2_expression_auc_oof_predictions.csv",
                "expression_ridge",
                "auc",
            ),
        },
    )
    summary = {
        "status": "completed",
        "experiment": "matched five-fold controlled model ablations",
        "variants": {
            "full_graph": "RGCN drug-target evidence encoder + expression encoder + pair head",
            "drug_id_only": "learned drug identity embedding + expression encoder; no graph propagation",
            "expression_only": "expression encoder only; no drug identity or graph",
        },
        "scope": "GDSC2 measured in-vitro single-agent response for known drugs and exact-name-matched cell lines",
        "outer_split": "frozen five-fold GroupKFold by Sanger model ID; folds 1-5",
        "inner_validation": "Same GroupShuffleSplit by Sanger model ID shared by all variants within each outer fold; used only to select the training epoch",
        "final_refit": "After selecting the epoch, reinitialise each model and train for that epoch count on every response row and Sanger model in the complete outer-training fold; the outer-test fold remains untouched",
        "target_scaling": "Separate LN(IC50) and AUC mean/std scalers are fitted on all outer-training response rows during final refit; predictions are inverse-transformed to original units",
        "expression_preprocessing": "Gene variance selection, mean, and standard deviation are refitted using all outer-training cell lines only; preprocessing is not fitted on validation or outer-test cell lines during final refit",
        "baseline_training_scope": "Final GNN refit uses the complete outer-training fold, aligning training-set cell-line/response-row scope with the existing outer-fold baseline estimators",
        "response_edges_used_for_message_passing": False,
        "model_message_passing_layers": 3,
        "drug_evidence_coverage": evidence_coverage,
        "fold_metrics": fold_metrics,
        "overall_metrics": overall,
        "external_baseline_comparisons": baseline_comparison,
        "inputs": {
            "response_pairs_sha256": sha256_file(pair_path),
            "expression_features_sha256": sha256_file(expression_path),
            "typed_graph_sha256": sha256_file(graph_path),
        },
        "outputs": {
            "oof_predictions": str(prediction_path),
            "metrics": str(metrics_path),
        },
        "limitations": [
            "This is a known-drug, held-out-cell-line, transductive in-vitro benchmark.",
            "It does not test unseen-drug generalisation or patient outcomes.",
            "The drug-ID-only and full-graph models are both transductive with respect to the known drug set.",
            "External baseline comparisons report overlap coverage; compare only where their curve IDs and frozen folds match.",
        ],
    }
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"overall_metrics": overall, "external_baseline_comparisons": baseline_comparison}, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--graph-file", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/ai/gdsc2_pair_gnn_ablations")
    parser.add_argument("--top-k-genes", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
