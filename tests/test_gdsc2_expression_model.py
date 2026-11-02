import pytest

from scripts.ai.train_gdsc2_expression_model import (
    _baseline_predictions,
    _build_exact_name_crosswalk,
    _fit_fold_models,
    _select_fold_features,
)
from scripts.ai import train_gdsc2_expression_model
from src.inference.gdsc2_expression import predict_expression_response


def _bundle():
    return {
        "schema_version": 1,
        "targets": {
            "ln_ic50": {
                "10": {
                    "feature_ids": ["ENSG1", "ENSG2"],
                    "feature_means": [1.0, 2.0],
                    "feature_scales": [1.0, 2.0],
                    "coefficients": [2.0, -1.0],
                    "intercept": -0.5,
                    "training_models": 25,
                }
            },
            "auc": {
                "10": {
                    "feature_ids": ["ENSG1", "ENSG2"],
                    "feature_means": [1.0, 2.0],
                    "feature_scales": [1.0, 2.0],
                    "coefficients": [0.5, 0.25],
                    "intercept": 0.4,
                    "training_models": 25,
                }
            },
        },
    }


def test_predict_expression_response_returns_each_trained_endpoint():
    result = predict_expression_response(
        _bundle(),
        "10",
        {"ENSG1": 3.0, "ENSG2": 3.0},
    )

    assert result["drug_id"] == "10"
    assert set(result["predictions"]) == {"ln_ic50", "auc"}
    assert result["training_models"] == {"ln_ic50": 25, "auc": 25}


def test_predict_expression_response_rejects_missing_features():
    with pytest.raises(ValueError, match="missing 1 required model features"):
        predict_expression_response(_bundle(), "10", {"ENSG1": 1.0})


def test_predict_expression_response_rejects_invalid_tpm():
    with pytest.raises(ValueError, match="finite and non-negative"):
        predict_expression_response(
            _bundle(),
            "10",
            {"ENSG1": -1.0, "ENSG2": 1.0},
        )


def test_predict_expression_response_does_not_fallback_for_unknown_drug():
    with pytest.raises(KeyError, match="No trained expression model"):
        predict_expression_response(_bundle(), "unknown", {"ENSG1": 1.0})


def test_exact_name_crosswalk_excludes_ambiguous_names():
    import pandas as pd

    responses = pd.DataFrame(
        {
            "sanger_model_id": ["SIDM1", "SIDM2", "SIDM3"],
            "cell_line": ["Line A", "Line B", "Line A"],
        }
    )
    crosswalk, audit = _build_exact_name_crosswalk(
        responses,
        ["line a, neuroblastoma", "Line B, other", "Line B, another"],
    )

    assert crosswalk == {}
    assert audit["ambiguous_names"] == 2
    assert audit["unmatched_models"] == 3


def test_fold_feature_selection_uses_training_columns_only():
    import numpy as np

    expression = np.asarray(
        [
            [0.0, 1.0, 0.0, 100.0],
            [4.0, 4.0, 4.0, -100.0],
            [0.0, 0.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    selected, means, scales = _select_fold_features(
        expression, [0, 1, 2], n_features=2
    )

    assert selected.tolist() == [2, 0]
    assert means.tolist() == pytest.approx([2 / 3, 1 / 3])
    assert scales.tolist() == pytest.approx([8**0.5 / 3, 2**0.5 / 3])


def test_drug_cancer_baseline_falls_back_when_training_group_is_small():
    import pandas as pd

    train = pd.DataFrame(
        {
            "drug_id": ["10"] * 20,
            "reported_cancer_type": ["NB"] * 9 + ["MB"] * 11,
            "ln_ic50": list(range(9)) + list(range(100, 111)),
        }
    )
    test = pd.DataFrame(
        {
            "drug_id": ["10"],
            "reported_cancer_type": ["NB"],
            "ln_ic50": [0.0],
        },
        index=[50],
    )

    result = _baseline_predictions(train, test, "ln_ic50")

    assert result[50]["baseline_drug_cancer_median"] == 100.5
    assert result[50]["baseline_drug_median"] == 100.5


def test_fold_model_predicts_only_held_out_rows():
    import numpy as np
    import pandas as pd

    model_ids = [f"SIDM{i:03}" for i in range(32)]
    frame = pd.DataFrame(
        {
            "drug_id": ["10"] * 32,
            "sanger_model_id": model_ids,
            "fold": [1] * 30 + [2] * 2,
            "ln_ic50": [float(i) for i in range(32)],
        }
    )
    features = np.asarray([[float(i), float(i % 5)] for i in range(32)])
    model_to_row = {model_id: index for index, model_id in enumerate(model_ids)}

    predictions = _fit_fold_models(
        frame,
        features,
        model_to_row,
        "ln_ic50",
        fold=2,
    )

    assert set(predictions) == {30, 31}
    assert all(np.isfinite(value) for value in predictions.values())


def test_training_pipeline_writes_oof_predictions_and_bundle(tmp_path, monkeypatch):
    import csv
    import json
    import numpy as np
    import pandas as pd

    responses_path = tmp_path / "responses.csv"
    folds_path = tmp_path / "folds.csv"
    output_dir = tmp_path / "models"
    rows = []
    fold_rows = []
    for model_index in range(30):
        for drug_id in ("10", "20"):
            curve_id = f"{drug_id}-{model_index}"
            rows.append(
                {
                    "curve_id": curve_id,
                    "drug_id": drug_id,
                    "drug_name": f"Drug {drug_id}",
                    "cell_line": f"Cell{model_index}",
                    "sanger_model_id": f"SIDM{model_index:03}",
                    "reported_cancer_type": "neuroblastoma",
                    "ln_ic50": float(model_index + int(drug_id)),
                    "auc": float(model_index) / 30,
                }
            )
            fold_rows.append(
                {"curve_id": curve_id, "fold": model_index % 5 + 1}
            )
    pd.DataFrame(rows).to_csv(responses_path, index=False)
    with folds_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["curve_id", "fold"])
        writer.writeheader()
        writer.writerows(fold_rows)

    labels = [f"Cell{index}, neuroblastoma" for index in range(30)]
    expression = np.asarray(
        [
            np.arange(30, dtype=np.float32),
            np.arange(30, dtype=np.float32) % 3,
            (np.arange(30, dtype=np.float32) ** 2) % 11,
            np.ones(30, dtype=np.float32),
        ]
    )
    monkeypatch.setattr(train_gdsc2_expression_model, "RESPONSE_SOURCE", responses_path)
    monkeypatch.setattr(train_gdsc2_expression_model, "FOLD_SOURCE", folds_path)
    monkeypatch.setattr(train_gdsc2_expression_model, "ROOT", tmp_path)
    monkeypatch.setattr(train_gdsc2_expression_model, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(
        train_gdsc2_expression_model,
        "MODEL_OUTPUT",
        output_dir / "model.json",
    )
    monkeypatch.setattr(
        train_gdsc2_expression_model,
        "SUMMARY_OUTPUT",
        output_dir / "summary.json",
    )
    monkeypatch.setattr(
        train_gdsc2_expression_model,
        "PREDICTION_OUTPUTS",
        {
            "ln_ic50": output_dir / "ln_ic50.csv",
            "auc": output_dir / "auc.csv",
        },
    )
    monkeypatch.setattr(train_gdsc2_expression_model, "N_FEATURES", 3)
    monkeypatch.setattr(train_gdsc2_expression_model, "MIN_TRAIN_MODELS", 8)
    monkeypatch.setattr(
        train_gdsc2_expression_model,
        "_read_expression_matrix",
        lambda: (labels, ["ENSG1", "ENSG2", "ENSG3", "ENSG4"], expression, "digest"),
    )

    train_gdsc2_expression_model.main()

    summary = json.loads((output_dir / "summary.json").read_text())
    bundle = json.loads((output_dir / "model.json").read_text())
    ln_ic50 = pd.read_csv(output_dir / "ln_ic50.csv")
    auc = pd.read_csv(output_dir / "auc.csv")
    assert summary["crosswalk"]["matched_models"] == 30
    assert summary["matched_drugs"] == 2
    assert ln_ic50["expression_ridge"].notna().all()
    assert auc["expression_ridge"].notna().all()
    assert set(bundle["targets"]) == {"ln_ic50", "auc"}
    assert set(bundle["targets"]["ln_ic50"]) == {"10", "20"}
