import pytest
import numpy as np
import pandas as pd
import torch

from scripts.ai.run_gdsc2_pair_gnn_ablations import (
    compare_external_baselines,
    fit_target_scaler,
    prepare_full_outer_refit,
    run_variant,
    split_outer_fold_rows,
    validate_full_outer_refit_partition,
    validate_pair_expression_alignment,
)
from scripts.ai.train_gdsc2_pair_gnn import select_and_scale_expression
from src.training.gdsc2_pair_ablations import (
    DrugIdExpressionModel,
    ExpressionOnlyModel,
)
from src.training.gdsc2_pair_gnn import build_compact_drug_graph
from torch_geometric.data import HeteroData
from types import SimpleNamespace


def test_target_scaler_is_endpoint_specific_and_invertible():
    targets = np.array(
        [[1.0, 0.1], [3.0, 0.3], [5.0, 0.5]],
        dtype=np.float32,
    )
    mean, scale = fit_target_scaler(targets)
    assert np.allclose(mean, [3.0, 0.3])
    assert np.allclose(scale, np.std(targets, axis=0))
    normalized = (targets - mean) / scale
    restored = normalized * scale + mean
    assert np.allclose(restored, targets)


def test_drug_id_expression_model_forward():
    model = DrugIdExpressionModel(
        num_drugs=4,
        num_expression_features=12,
        hidden_dim=16,
    )
    predictions = model(
        torch.randn(5, 12),
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 2, 4]),
    )
    assert predictions.shape == (3, 2)
    assert torch.isfinite(predictions).all()


def test_expression_only_model_ignores_drug_identity():
    model = ExpressionOnlyModel(
        num_expression_features=12,
        hidden_dim=16,
        dropout=0.0,
    )
    model.eval()
    expression = torch.randn(5, 12)
    cells = torch.tensor([0, 2, 4])
    first = model(expression, torch.tensor([0, 1, 2]), cells)
    second = model(expression, torch.tensor([3, 2, 1]), cells)
    assert torch.allclose(first, second)
    assert first.shape == (3, 2)



def test_expression_feature_selection_uses_inner_fit_cell_lines_only():
    model_ids = np.asarray(["A", "B", "C", "D", "E"])
    expression = np.asarray(
        [
            [1.0, 5.0, 2.0, 7.0],
            [2.0, 4.0, 3.0, 6.0],
            [4.0, 1.0, 5.0, 2.0],
            [10.0, 100.0, 1.0, 50.0],
            [11.0, 200.0, 2.0, 60.0],
        ],
        dtype=np.float32,
    )
    train_ids = {"A", "B", "C"}
    first_scaled, first_meta = select_and_scale_expression(
        expression, model_ids, train_ids, top_k=3
    )

    changed_heldout = expression.copy()
    changed_heldout[3:] = np.asarray(
        [[1e6, -1e6, 2e6, -2e6], [-3e6, 3e6, -4e6, 4e6]],
        dtype=np.float32,
    )
    second_scaled, second_meta = select_and_scale_expression(
        changed_heldout, model_ids, train_ids, top_k=3
    )

    assert first_meta["fit_scope"] == "inner-fit cell lines only"
    assert first_meta["training_cell_lines_for_scaler"] == 3
    assert first_meta["selected_gene_indices"] == second_meta["selected_gene_indices"]
    assert np.allclose(first_meta["scaler_mean"], second_meta["scaler_mean"])
    assert np.allclose(first_meta["scaler_scale"], second_meta["scaler_scale"])
    assert np.allclose(first_scaled[:3], second_scaled[:3])


def _pair_rows_for_fold_test():
    rows = []
    for fold in range(1, 6):
        for model_index in range(2):
            sid = f"SIDM_{fold}_{model_index}"
            for drug in ("1", "2"):
                rows.append({
                    "curve_id": f"{sid}_{drug}",
                    "drug_id": drug,
                    "sanger_model_id": sid,
                    "graph_node_id": f"GDSC_MODEL:{sid}",
                    "expression_row_index": 2 * (fold - 1) + model_index,
                    "fold": fold,
                    "ln_ic50": 1.0,
                    "auc": 0.5,
                })
    return pd.DataFrame(rows)


def test_pair_expression_alignment_rejects_wrong_expression_row():
    pairs = _pair_rows_for_fold_test()
    model_ids = np.asarray([
        f"SIDM_{fold}_{index}"
        for fold in range(1, 6)
        for index in range(2)
    ])
    validate_pair_expression_alignment(pairs, model_ids)
    pairs.loc[0, "expression_row_index"] = 1
    with pytest.raises(ValueError, match="expression_row_index"):
        validate_pair_expression_alignment(pairs, model_ids)


def test_pair_expression_alignment_rejects_mismatched_graph_node_id():
    pairs = _pair_rows_for_fold_test()
    model_ids = np.asarray([
        f"SIDM_{fold}_{index}"
        for fold in range(1, 6)
        for index in range(2)
    ])
    pairs.loc[0, "graph_node_id"] = "GDSC_MODEL:WRONG"
    with pytest.raises(ValueError, match="graph_node_id"):
        validate_pair_expression_alignment(pairs, model_ids)


def test_inner_fit_validation_and_outer_test_groups_are_disjoint():
    pairs = _pair_rows_for_fold_test()
    fit_rows, val_rows, test_rows = split_outer_fold_rows(
        pairs, fold=1, validation_fraction=0.25, seed=51
    )
    fit_ids = set(fit_rows["sanger_model_id"])
    val_ids = set(val_rows["sanger_model_id"])
    test_ids = set(test_rows["sanger_model_id"])
    assert fit_ids.isdisjoint(val_ids)
    assert fit_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert fit_ids | val_ids | test_ids == set(pairs["sanger_model_id"])
    assert set(test_rows["fold"]) == {1}
    assert set(fit_rows["fold"]) | set(val_rows["fold"]) == {2, 3, 4, 5}


def test_fold_split_rejects_sanger_model_crossing_outer_folds():
    pairs = _pair_rows_for_fold_test()
    pairs.loc[1, "fold"] = 2
    with pytest.raises(ValueError, match="multiple outer folds"):
        split_outer_fold_rows(pairs, fold=1, validation_fraction=0.25, seed=51)



def test_external_baseline_comparison_checks_cell_line_identity(tmp_path):
    baseline_path = tmp_path / "baseline.csv"
    rows = []
    oof = []
    for index in range(4):
        sid = f"SIDM{index}"
        fold = (index % 2) + 1
        actual = 1.0 + index
        oof_row = {
            "curve_id": f"CURVE{index}",
            "fold": fold,
            "sanger_model_id": sid,
            "actual_ln_ic50": actual,
            "pred_full_graph_ln_ic50": actual + 0.1,
            "pred_drug_id_only_ln_ic50": actual + 0.2,
            "pred_expression_only_ln_ic50": actual + 0.3,
        }
        oof.append(oof_row)
        rows.append({
            "curve_id": f"CURVE{index}",
            "fold": fold,
            "sanger_model_id": sid,
            "actual_ln_ic50": actual,
            "B2_drug_cancer_median": actual + 0.15,
        })
    baseline = pd.DataFrame(rows)
    baseline.to_csv(baseline_path, index=False)

    comparison = compare_external_baselines(
        pd.DataFrame(oof),
        {
            "drug_cancer_median_ln_ic50": (
                baseline_path, "B2_drug_cancer_median", "actual_ln_ic50"
            )
        },
    )
    result = comparison["drug_cancer_median_ln_ic50_ln_ic50"]
    assert result["status"] == "compared"
    assert result["overlap_rows"] == 4
    assert result["fold_assignments_identical"] is True
    assert result["sanger_model_ids_identical"] is True
    assert result["measured_targets_identical"] is True

    baseline.loc[0, "sanger_model_id"] = "WRONG"
    baseline.to_csv(baseline_path, index=False)
    with pytest.raises(ValueError, match="Sanger model IDs differ"):
        compare_external_baselines(
            pd.DataFrame(oof),
            {
                "drug_cancer_median_ln_ic50": (
                    baseline_path, "B2_drug_cancer_median", "actual_ln_ic50"
                )
            },
        )



def test_full_outer_refit_uses_all_outer_training_cell_lines_and_rows():
    pairs = _pair_rows_for_fold_test()
    fit_rows, val_rows, test_rows = split_outer_fold_rows(
        pairs, fold=1, validation_fraction=0.25, seed=91
    )
    outer_train = pairs[pairs["fold"] != 1].copy()
    train_groups = validate_full_outer_refit_partition(
        fit_rows, val_rows, outer_train, test_rows
    )
    assert train_groups == set(outer_train["sanger_model_id"])
    assert len(train_groups) == outer_train["sanger_model_id"].nunique()
    assert train_groups.isdisjoint(set(test_rows["sanger_model_id"]))


def test_full_outer_refit_preprocessing_uses_all_train_rows_not_test_rows():
    pairs = _pair_rows_for_fold_test()
    fit_rows, val_rows, test_rows = split_outer_fold_rows(
        pairs, fold=1, validation_fraction=0.25, seed=92
    )
    outer_train = pairs[pairs["fold"] != 1].copy()
    model_ids = np.asarray([
        f"SIDM_{fold}_{index}"
        for fold in range(1, 6)
        for index in range(2)
    ])
    rng = np.random.default_rng(19)
    expression = rng.normal(size=(10, 8)).astype(np.float32)

    first = prepare_full_outer_refit(
        expression, model_ids, outer_train, test_rows, top_k_genes=5
    )
    changed_expression = expression.copy()
    test_ids = set(test_rows["sanger_model_id"])
    for index, model_id in enumerate(model_ids):
        if model_id in test_ids:
            changed_expression[index] = np.linspace(
                -1e6, 1e6, changed_expression.shape[1]
            )
    changed_test_rows = test_rows.copy()
    changed_test_rows["ln_ic50"] = 1e9
    changed_test_rows["auc"] = -1e9
    second = prepare_full_outer_refit(
        changed_expression, model_ids, outer_train, changed_test_rows,
        top_k_genes=5,
    )

    first_expression, first_meta, first_mean, first_scale = first
    second_expression, second_meta, second_mean, second_scale = second
    assert first_meta["fit_scope"] == "complete outer-training cell lines"
    assert first_meta["training_cell_lines_for_scaler"] == outer_train["sanger_model_id"].nunique()
    assert first_meta["selected_gene_indices"] == second_meta["selected_gene_indices"]
    assert np.allclose(first_meta["scaler_mean"], second_meta["scaler_mean"])
    assert np.allclose(first_meta["scaler_scale"], second_meta["scaler_scale"])
    assert np.allclose(first_mean, second_mean)
    assert np.allclose(first_scale, second_scale)
    outer_train_ids = set(outer_train["sanger_model_id"])
    train_expression_rows = [
        index for index, model_id in enumerate(model_ids)
        if model_id in outer_train_ids
    ]
    assert np.allclose(
        first_expression[train_expression_rows],
        second_expression[train_expression_rows],
    )



def _tiny_response_free_graph():
    graph = HeteroData()
    graph["drug"].node_ids = ["GDSC_DRUG:1", "GDSC_DRUG:2"]
    graph["drug"].num_nodes = 2
    graph["chembl_molecule"].node_ids = ["CHEMBL:1", "CHEMBL:2"]
    graph["chembl_molecule"].num_nodes = 2
    graph["target"].node_ids = ["TARGET:1", "TARGET:2"]
    graph["target"].num_nodes = 2
    graph["gene"].node_ids = ["GENE:1", "GENE:2"]
    graph["gene"].num_nodes = 2
    graph["cell_line"].node_ids = [f"GDSC_MODEL:SIDM_{i}" for i in range(1, 6)]
    graph["cell_line"].num_nodes = 5
    graph["drug", "exact_name_chembl_mapping", "chembl_molecule"].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )
    graph["chembl_molecule", "chembl_direct_mechanism", "target"].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )
    graph["target", "has_chembl_gene_symbol", "gene"].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )
    # A response edge is intentionally present but the compact builder drops it.
    graph["drug", "gdsc2_response", "cell_line"].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )
    graph["drug", "gdsc2_response", "cell_line"].ln_ic50 = torch.tensor(
        [99.0, 99.0], dtype=torch.float32
    )
    return build_compact_drug_graph(graph)


def test_final_refit_checkpoint_is_independent_of_outer_test_labels():
    rows = []
    for fold in range(1, 6):
        sid = f"SIDM_{fold}"
        for drug_id, offset in (("1", 0.0), ("2", 0.4)):
            rows.append({
                "curve_id": f"{sid}_{drug_id}",
                "drug_id": drug_id,
                "sanger_model_id": sid,
                "fold": fold,
                "ln_ic50": float(fold + offset),
                "auc": float(0.1 * fold + 0.01 * offset),
            })
    pairs = pd.DataFrame(rows)
    fit_rows, val_rows, test_rows = split_outer_fold_rows(
        pairs, fold=1, validation_fraction=0.25, seed=13
    )
    outer_train_rows = pairs[pairs["fold"] != 1].copy()
    model_ids = np.asarray([f"SIDM_{i}" for i in range(1, 6)])
    expression = np.random.default_rng(21).normal(size=(5, 8)).astype(np.float32)
    fit_groups = set(fit_rows["sanger_model_id"])
    selection_expression, selection_meta = select_and_scale_expression(
        expression, model_ids, fit_groups, top_k=4
    )
    graph = _tiny_response_free_graph()
    drug_map = {"1": 0, "2": 1}
    args = SimpleNamespace(
        seed=7,
        hidden_dim=8,
        dropout=0.0,
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=2,
        patience=2,
        min_delta=0.0,
        top_k_genes=4,
    )

    first_result, first_metrics, first_checkpoint = run_variant(
        "full_graph", 1, fit_rows, val_rows, outer_train_rows, test_rows,
        expression, selection_expression, model_ids, graph, drug_map,
        selection_meta, args, torch.device("cpu"),
    )
    changed_test = test_rows.copy()
    changed_test["ln_ic50"] = changed_test["ln_ic50"] + 1e6
    changed_test["auc"] = changed_test["auc"] - 1e6
    _, second_metrics, second_checkpoint = run_variant(
        "full_graph", 1, fit_rows, val_rows, outer_train_rows, changed_test,
        expression, selection_expression, model_ids, graph, drug_map,
        selection_meta, args, torch.device("cpu"),
    )

    assert first_metrics["final_refit_used_all_outer_training_rows"] is True
    assert first_metrics["outer_training_cell_lines"] == 4
    assert first_metrics["outer_training_response_rows"] == len(outer_train_rows)
    assert first_metrics["final_refit_epochs"] == first_metrics["selected_epoch_from_inner_validation"]
    assert second_metrics["selected_epoch_from_inner_validation"] == first_metrics["selected_epoch_from_inner_validation"]
    assert first_checkpoint["training_scope"] == "all outer-training Sanger model IDs and response rows"
    assert first_checkpoint["selected_gene_indices"] == second_checkpoint["selected_gene_indices"]
    assert np.allclose(first_checkpoint["target_mean"], second_checkpoint["target_mean"])
    for key, value in first_checkpoint["state_dict"].items():
        assert torch.equal(value, second_checkpoint["state_dict"][key]), key
