import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "ai" / "run_gdsc2_pair_gnn_multiseed.py"
)
SPEC = importlib.util.spec_from_file_location("gdsc2_multiseed", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def _oof_frame():
    rows = []
    for fold in range(1, 6):
        for index in range(2):
            rows.append({
                "curve_id": f"{fold}-{index}",
                "drug_id": "1",
                "sanger_model_id": f"SIDM{fold}-{index}",
                "fold": fold,
                "actual_ln_ic50": float(fold + index),
                "actual_auc": float((fold + index) / 10),
                "pred_full_graph_ln_ic50": 0.0,
                "pred_full_graph_auc": 0.0,
                "pred_drug_id_only_ln_ic50": 0.0,
                "pred_drug_id_only_auc": 0.0,
                "pred_expression_only_ln_ic50": 0.0,
                "pred_expression_only_auc": 0.0,
            })
    return pd.DataFrame(rows)


def test_oof_file_validator_accepts_valid_five_fold_predictions(tmp_path):
    path = tmp_path / "oof.csv"
    frame = _oof_frame()
    frame.to_csv(path, index=False)
    loaded = MODULE.validate_oof_file(path)
    assert len(loaded) == 10
    assert set(loaded["fold"]) == {1, 2, 3, 4, 5}


def test_oof_file_validator_rejects_changed_curve_ids_between_seeds(tmp_path):
    reference = _oof_frame()
    other = reference.copy()
    other.loc[0, "curve_id"] = "changed"
    path = tmp_path / "other.csv"
    other.to_csv(path, index=False)
    with pytest.raises(ValueError, match="curve IDs differ"):
        MODULE.validate_oof_file(path, reference)


def test_oof_file_validator_rejects_changed_fold_assignments(tmp_path):
    reference = _oof_frame()
    other = reference.copy()
    other.loc[0, "fold"] = 2
    path = tmp_path / "other.csv"
    other.to_csv(path, index=False)
    with pytest.raises(ValueError, match="fold assignments differ|frozen folds 1-5|assigns a Sanger model"):
        MODULE.validate_oof_file(path, reference)


def test_oof_file_validator_rejects_nonfinite_predictions(tmp_path):
    frame = _oof_frame()
    frame.loc[0, "pred_full_graph_auc"] = np.nan
    path = tmp_path / "oof.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-finite"):
        MODULE.validate_oof_file(path)



def test_fresh_run_set_ignores_existing_two_layer_reference(tmp_path):
    old_reference = tmp_path / "old_two_layer_oof.csv"
    old_reference.write_text("this file must not be read in fresh mode", encoding="utf-8")
    existing_runs, reference_frame = MODULE.resolve_reference_run(
        old_reference,
        fresh_run_set=True,
        require_reference=False,
    )
    assert existing_runs == []
    assert reference_frame is None


def test_fresh_run_set_cannot_require_reference():
    with pytest.raises(ValueError, match="cannot be combined"):
        MODULE.resolve_reference_run(
            Path("unused.csv"),
            fresh_run_set=True,
            require_reference=True,
        )
