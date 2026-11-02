import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ai" / "analyze_gdsc2_pair_gnn_uncertainty.py"
SPEC = importlib.util.spec_from_file_location("gdsc2_uncertainty", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def _frame():
    rows = []
    for fold in range(1, 6):
        for model in range(3):
            sid = f"SIDM{fold}_{model}"
            for drug in range(2):
                actual = float(fold + model + drug)
                rows.append({
                    "sanger_model_id": sid,
                    "fold": fold,
                    "actual": actual,
                    "pred_a": actual + 0.1,
                    "pred_b": actual + 0.3,
                })
    return pd.DataFrame(rows)


def test_cluster_bootstrap_is_paired_and_reproducible():
    frame = _frame()
    first = MODULE.cluster_bootstrap_mae_difference(
        frame, "actual", "pred_a", "pred_b", n_bootstrap=500, seed=7
    )
    second = MODULE.cluster_bootstrap_mae_difference(
        frame, "actual", "pred_a", "pred_b", n_bootstrap=500, seed=7
    )
    assert first == second
    assert first["point_mae_difference_a_minus_b"] == pytest.approx(-0.2)
    assert first["ci95_upper"] < 0
    assert first["n_cell_lines"] == 15
    assert first["n_response_rows"] == 30


def test_cluster_bootstrap_rejects_cell_line_in_multiple_folds():
    frame = _frame()
    frame.loc[1, "fold"] = 2
    with pytest.raises(ValueError, match="more than one outer fold"):
        MODULE.cluster_bootstrap_mae_difference(
            frame, "actual", "pred_a", "pred_b", n_bootstrap=100
        )


def test_cluster_bootstrap_rejects_missing_predictions():
    frame = _frame()
    frame.loc[0, "pred_a"] = np.nan
    with pytest.raises(ValueError, match="missing cluster, fold, target, or prediction"):
        MODULE.cluster_bootstrap_mae_difference(
            frame, "actual", "pred_a", "pred_b", n_bootstrap=100
        )


def test_cluster_bootstrap_rejects_too_few_replicates():
    with pytest.raises(ValueError, match="at least 100"):
        MODULE.cluster_bootstrap_mae_difference(
            _frame(), "actual", "pred_a", "pred_b", n_bootstrap=50
        )
