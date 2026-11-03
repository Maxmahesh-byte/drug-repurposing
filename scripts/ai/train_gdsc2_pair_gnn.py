#!/usr/bin/env python3
"""Train and evaluate the five-fold GDSC2 drug/cell-line pair GNN.

The fixed GDSC2 folds are held out by Sanger model ID. The drug evidence graph
contains only drug/ChEMBL/target/gene relations; response edges and response
attributes are never loaded. Expression feature selection and scaling are fit
on training cell lines only in each outer fold.
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

from src.training.gdsc2_pair_gnn import DrugCellLinePairGNN, build_compact_drug_graph

DEFAULT_DATA = ROOT / "data/ai/response_dataset"
DEFAULT_GRAPH = ROOT / "data/processed/graph_v2/typed_graph_v2.pt"
DEFAULT_OUTPUT = ROOT / "data/ai/gdsc2_pair_gnn"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_and_scale_expression(
    expression: np.ndarray,
    model_ids: np.ndarray,
    train_model_ids: set[str],
    top_k: int,
) -> tuple[np.ndarray, dict]:
    train_indices = np.asarray(
        [i for i, model_id in enumerate(model_ids.astype(str)) if model_id in train_model_ids],
        dtype=np.int64,
    )
    if len(train_indices) < 3:
        raise ValueError("Too few training cell lines for fold-specific expression selection.")
    train_values = expression[train_indices].astype(np.float64, copy=False)
    variances = np.var(train_values, axis=0, ddof=1)
    finite = np.isfinite(variances)
    candidates = np.flatnonzero(finite)
    if len(candidates) == 0:
        raise ValueError("No finite expression features available.")
    k = min(int(top_k), len(candidates))
    ranked = candidates[np.argsort(variances[candidates], kind="mergesort")[-k:]]
    mean = train_values[:, ranked].mean(axis=0)
    scale = train_values[:, ranked].std(axis=0, ddof=0)
    scale[scale < 1e-8] = 1.0
    transformed = (expression[:, ranked].astype(np.float32) - mean.astype(np.float32)) / scale.astype(np.float32)
    transformed = np.clip(transformed, -8.0, 8.0)
    metadata = {
        "selected_feature_count": int(k),
        "selected_gene_indices": ranked.astype(int).tolist(),
        "training_cell_lines_for_scaler": int(len(train_indices)),
        "fit_scope": "inner-fit cell lines only",
        "scaler_mean": mean.astype(float).tolist(),
        "scaler_scale": scale.astype(float).tolist(),
    }
    return transformed.astype(np.float32), metadata


def metric_block(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
    }


def run_fold(
    fold: int,
    pairs: pd.DataFrame,
    expression: np.ndarray,
    model_ids: np.ndarray,
    gene_ids: np.ndarray,
    drug_graph,
    drug_id_to_index: dict[str, int],
    args,
    device: torch.device,
) -> tuple[pd.DataFrame, dict]:
    train_rows = pairs[pairs["fold"] != fold].copy()
    test_rows = pairs[pairs["fold"] == fold].copy()
    if train_rows.empty or test_rows.empty:
        raise ValueError(f"Fold {fold} has empty train or test rows.")

    # A group-level validation split is carved only from the outer training set.
    group_split = GroupShuffleSplit(n_splits=1, test_size=args.validation_fraction, random_state=args.seed + fold)
    train_idx, val_idx = next(
        group_split.split(train_rows, groups=train_rows["sanger_model_id"])
    )
    fit_rows = train_rows.iloc[train_idx].copy()
    val_rows = train_rows.iloc[val_idx].copy()
    fit_groups = set(fit_rows["sanger_model_id"].astype(str))
    val_groups = set(val_rows["sanger_model_id"].astype(str))
    if fit_groups.intersection(val_groups):
        raise RuntimeError("Inner validation split leaks cell lines.")

    scaled_expression, feature_meta = select_and_scale_expression(
        expression, model_ids, fit_groups, args.top_k_genes
    )
    drug_ids = pairs["drug_id"].astype(str).unique()
    unknown_drugs = sorted(set(drug_ids) - set(drug_id_to_index))
    if unknown_drugs:
        raise ValueError(f"Response drugs missing graph nodes: {unknown_drugs[:10]}")

    model = DrugCellLinePairGNN(
        num_graph_nodes=drug_graph.num_nodes,
        num_node_types=len(drug_graph.node_counts),
        num_relations=drug_graph.num_relations,
        num_drugs=len(drug_graph.drug_ids),
        num_expression_features=scaled_expression.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    edge_index = drug_graph.edge_index.to(device)
    edge_type = drug_graph.edge_type.to(device)
    node_type = drug_graph.node_type.to(device)
    expression_tensor = torch.from_numpy(scaled_expression).to(device)
    model_row_lookup = {str(model_id): i for i, model_id in enumerate(model_ids.astype(str))}

    def encode_and_predict(frame: pd.DataFrame) -> torch.Tensor:
        drug_embeddings, cell_embeddings = model.encode(
            expression_tensor, edge_index, edge_type, node_type
        )
        drug_index = torch.tensor(
            [drug_id_to_index[str(value)] for value in frame["drug_id"]],
            dtype=torch.long, device=device,
        )
        cell_index = torch.tensor(
            [model_row_lookup[str(value)] for value in frame["sanger_model_id"]],
            dtype=torch.long, device=device,
        )
        return model.score_pairs(drug_embeddings, cell_embeddings, drug_index, cell_index)

    target_columns = ["ln_ic50", "auc"]
    fit_targets = torch.tensor(fit_rows[target_columns].to_numpy(dtype=np.float32), device=device)
    val_targets = torch.tensor(val_rows[target_columns].to_numpy(dtype=np.float32), device=device)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        fit_predictions = encode_and_predict(fit_rows)
        loss = torch.nn.functional.smooth_l1_loss(fit_predictions, fit_targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss in fold {fold}, epoch {epoch}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = encode_and_predict(val_rows)
            val_loss = torch.nn.functional.smooth_l1_loss(val_predictions, val_targets).item()
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = args.patience
        else:
            patience_left -= 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"fold={fold} epoch={epoch} train_smooth_l1={loss.item():.5f} "
                f"val_smooth_l1={val_loss:.5f}",
                flush=True,
            )
        if patience_left <= 0:
            break

    if best_state is None:
        raise RuntimeError(f"No best model state was captured for fold {fold}.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = encode_and_predict(test_rows).cpu().numpy()

    result = test_rows[["curve_id", "drug_id", "sanger_model_id", "fold", "ln_ic50", "auc"]].copy()
    result["pred_ln_ic50"] = predictions[:, 0]
    result["pred_auc"] = predictions[:, 1]
    metrics = {
        "fold": fold,
        "fit_cell_lines": len(fit_groups),
        "inner_validation_cell_lines": len(val_groups),
        "outer_test_cell_lines": int(test_rows["sanger_model_id"].nunique()),
        "fit_response_rows": int(len(fit_rows)),
        "inner_validation_rows": int(len(val_rows)),
        "outer_test_rows": int(len(test_rows)),
        "best_epoch": best_epoch,
        "best_inner_validation_smooth_l1": best_val,
        "expression_preprocessing": feature_meta,
        "ln_ic50": metric_block(test_rows["ln_ic50"].to_numpy(), predictions[:, 0]),
        "auc": metric_block(test_rows["auc"].to_numpy(), predictions[:, 1]),
    }
    return result, metrics, best_state


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

    print("Loading prepared response pairs and expression features...", flush=True)
    pairs = pd.read_csv(
        pair_path,
        dtype={"curve_id": str, "drug_id": str, "sanger_model_id": str},
    )
    with np.load(expression_path, allow_pickle=False) as archive:
        expression = archive["expression_log2_tpm"].astype(np.float32)
        gene_ids = archive["gene_ids"].astype(str)
        model_ids = archive["sanger_model_ids"].astype(str)
    if expression.shape != (len(model_ids), len(gene_ids)):
        raise ValueError("Expression feature matrix dimensions do not align with metadata.")
    if pairs["curve_id"].duplicated().any():
        raise ValueError("Response-pair table contains duplicate curve IDs.")
    if set(pairs["fold"].unique()) != set(range(1, 6)):
        raise ValueError("Expected frozen fold IDs 1–5.")

    print("Loading typed graph and extracting response-free drug evidence subgraph...", flush=True)
    # Only load the trusted locally generated graph artifact.
    typed_graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    compact_graph = build_compact_drug_graph(typed_graph)
    drug_id_to_index = {
        drug_node.split(":", 1)[1]: index
        for index, drug_node in enumerate(compact_graph.drug_ids)
        if drug_node.startswith("GDSC_DRUG:")
    }
    if len(drug_id_to_index) != len(compact_graph.drug_ids):
        raise ValueError("Unexpected drug node ID format in typed graph.")
    graph_drugs = set(drug_id_to_index)
    response_drugs = set(pairs["drug_id"].astype(str))
    missing = sorted(response_drugs - graph_drugs)
    if missing:
        raise ValueError(
            f"{len(missing)} response drugs are missing graph drug nodes; "
            f"examples: {missing[:10]}"
        )

    # Verify every matched cell-line ID is in the graph, but do not add response
    # edges to the model graph or use their legacy edge attributes.
    cell_ids = set(pairs["graph_node_id"].astype(str))
    graph_cell_ids = set(str(value) for value in typed_graph["cell_line"].node_ids)
    missing_cells = sorted(cell_ids - graph_cell_ids)
    if missing_cells:
        raise ValueError(f"Matched response cell-line nodes missing from graph: {missing_cells[:10]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device={device}; drug graph nodes={compact_graph.num_nodes}; "
        f"drug graph edges={compact_graph.edge_index.shape[1]}; "
        f"response pairs={len(pairs)}",
        flush=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_predictions = []
    fold_metrics = []
    for fold in range(1, 6):
        print(f"\n===== Outer held-out fold {fold}/5 =====", flush=True)
        predictions, metrics, best_state = run_fold(
            fold, pairs, expression, model_ids, gene_ids,
            compact_graph, drug_id_to_index, args, device,
        )
        fold_predictions.append(predictions)
        fold_metrics.append(metrics)
        fold_path = output_dir / f"fold_{fold}_best_state.pt"
        torch.save(
            {
                "state_dict": best_state,
                "fold": fold,
                "model_config": {
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "top_k_genes": args.top_k_genes,
                },
                "drug_ids": compact_graph.drug_ids,
                "gene_ids": gene_ids.tolist(),
            },
            fold_path,
        )
        print(
            f"fold={fold} LN_IC50_MAE={metrics['ln_ic50']['mae']:.5f} "
            f"AUC_MAE={metrics['auc']['mae']:.5f}",
            flush=True,
        )

    oof = pd.concat(fold_predictions, ignore_index=True).sort_values("curve_id")
    oof_path = output_dir / "gdsc2_pair_gnn_oof_predictions.csv"
    metrics_path = output_dir / "gdsc2_pair_gnn_metrics.json"
    oof.to_csv(oof_path, index=False)

    # Fair baseline comparison on the exact same matched curve IDs and frozen folds.
    baseline_dir = ROOT / "data/ai/baselines"
    baseline_comparison = {}
    baseline_specs = {
        "ln_ic50": (
            baseline_dir / "gdsc2_ln_ic50_oof_predictions.csv",
            "actual_ln_ic50",
            "B2_drug_cancer_median",
            "pred_ln_ic50",
        ),
        "auc": (
            baseline_dir / "gdsc2_auc_oof_predictions.csv",
            "actual_auc",
            "B2_drug_cancer_median",
            "pred_auc",
        ),
    }
    for endpoint, (baseline_path, actual_col, baseline_col, gnn_col) in baseline_specs.items():
        if not baseline_path.is_file():
            baseline_comparison[endpoint] = {
                "status": "baseline_file_missing",
                "expected_path": str(baseline_path),
            }
            continue
        baseline = pd.read_csv(
            baseline_path,
            dtype={"curve_id": str, "drug_id": str, "sanger_model_id": str},
        )
        if baseline["curve_id"].duplicated().any():
            raise ValueError(f"Duplicate curve IDs in baseline file: {baseline_path}")
        matched = oof[["curve_id", "fold", gnn_col]].merge(
            baseline[["curve_id", "fold", actual_col, baseline_col]],
            on="curve_id",
            how="inner",
            validate="one_to_one",
            suffixes=("_gnn", "_baseline"),
        )
        if len(matched) != len(oof):
            raise ValueError(
                f"Baseline {endpoint} overlap {len(matched)} does not equal "
                f"GNN OOF rows {len(oof)}; cannot claim same-row comparison."
            )
        if not np.array_equal(
            matched["fold_gnn"].to_numpy(), matched["fold_baseline"].to_numpy()
        ):
            raise ValueError(f"Frozen fold assignments differ in {endpoint} baseline.")
        actual = matched[actual_col].to_numpy(dtype=float)
        baseline_pred = matched[baseline_col].to_numpy(dtype=float)
        gnn_pred = matched[gnn_col].to_numpy(dtype=float)
        baseline_comparison[endpoint] = {
            "same_curve_ids": int(len(matched)),
            "frozen_fold_assignments_identical": True,
            "drug_cancer_median_baseline": metric_block(actual, baseline_pred),
            "pair_gnn": metric_block(actual, gnn_pred),
            "mae_difference_gnn_minus_baseline": float(
                mean_absolute_error(actual, gnn_pred)
                - mean_absolute_error(actual, baseline_pred)
            ),
        }

    summary = {
        "status": "trained_and_evaluated",
        "model": "drug-side RGCN evidence encoder + fold-specific cell-line expression MLP + pairwise dual-endpoint head",
        "scope": "GDSC2 in-vitro single-agent response; held-out cell lines; known drugs",
        "outer_split": "existing five-fold GroupKFold by Sanger model ID; fold IDs 1-5",
        "inner_validation": "GroupShuffleSplit by Sanger model ID from outer training data only",
        "response_relation_used_for_message_passing": False,
        "response_edge_attributes_used_as_features": False,
        "fold_specific_expression_selection_and_scaling": True,
        "expression_feature_selection": f"top {args.top_k_genes} training-fold variance genes",
        "drug_graph": {
            "node_counts": compact_graph.node_counts,
            "relation_counts": compact_graph.relation_counts,
            "bidirectional_message_passing_relations": compact_graph.num_relations,
        },
        "fold_metrics": fold_metrics,
        "same_row_baseline_comparison": baseline_comparison,
        "overall": {
            endpoint: metric_block(
                oof[f"actual_{endpoint}"].to_numpy() if f"actual_{endpoint}" in oof else oof[endpoint].to_numpy(),
                oof[f"pred_{endpoint}"].to_numpy(),
            )
            for endpoint in ("ln_ic50", "auc")
        },
        "inputs": {
            "response_pairs_sha256": sha256_file(pair_path),
            "expression_features_sha256": sha256_file(expression_path),
            "typed_graph_sha256": sha256_file(graph_path),
        },
        "outputs": {
            "oof_predictions": str(oof_path),
            "metrics": str(metrics_path),
        },
        "limitations": [
            "This is an exploratory in-vitro cell-line benchmark, not a clinical response predictor.",
            "All drugs are known drugs in the dataset; this does not test unseen-drug generalization.",
            "Drug node embeddings are transductive learned parameters; the graph relations provide target-evidence context.",
            "No OpenPedCan patient outcome or patient-level label is used.",
        ],
    }
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--graph-file", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
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
