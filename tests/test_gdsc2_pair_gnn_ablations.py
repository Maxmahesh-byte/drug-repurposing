import pytest
import numpy as np
import pandas as pd
import torch

from scripts.ai.run_gdsc2_pair_gnn_ablations import (
    compare_external_baselines,
    fit_target_scaler,
    split_outer_fold_rows,
    validate_pair_expression_alignment,
)\nfrom scripts.ai.train_gdsc2_pair_gnn import select_and_scale_expression
from src.training.gdsc2_pair_ablations import (
    DrugIdExpressionModel,
    ExpressionOnlyModel,
)


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
