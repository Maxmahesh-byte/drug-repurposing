import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.training.train_almanac_baseline import (
    _grouped_oof,
    _load_scores,
    train_almanac_baseline,
)


def _scores():
    rows = []
    drugs = [f"Drug {index}" for index in range(6)]
    for cell_index in range(6):
        for drug_a_index in range(len(drugs)):
            for drug_b_index in range(drug_a_index + 1, len(drugs)):
                rows.append(
                    {
                        "cell_line": f"Cell {cell_index}",
                        "drug_a": drugs[drug_a_index],
                        "drug_b": drugs[drug_b_index],
                        "combo_score": (
                            cell_index * 3 + drug_a_index - drug_b_index
                        ),
                        "score_complete": True,
                    }
                )
    return pd.DataFrame(rows)


def test_load_scores_excludes_incomplete_labels_and_rejects_duplicate_rows(tmp_path):
    source = tmp_path / "scores.csv"
    scores = _scores()
    scores.loc[0, ["combo_score", "score_complete"]] = [None, False]
    scores.to_csv(source, index=False)

    loaded = _load_scores(source)

    assert len(loaded) == len(scores) - 1
    assert loaded["combo_score"].notna().all()
    assert loaded["pair_id"].nunique() == 15

    duplicate_source = tmp_path / "duplicates.csv"
    pd.concat([_scores(), _scores().iloc[[0]]], ignore_index=True).to_csv(
        duplicate_source,
        index=False,
    )
    with pytest.raises(ValueError, match="duplicate cell-line/drug-pair"):
        _load_scores(duplicate_source)


@pytest.mark.parametrize(
    ("group_column", "scenario"),
    [
        ("pair_id", "drug_pair_holdout"),
        ("cell_line", "cell_line_holdout"),
    ],
)
def test_grouped_oof_keeps_groups_inside_one_test_fold(group_column, scenario):
    scores = _scores()
    from src.training.train_almanac_baseline import _load_scores

    # The loader adds the stable canonical pair identifier used by the split.
    scores = _load_scores_from_frame(scores)
    oof, summary = _grouped_oof(scores, group_column, scenario)

    assert len(oof) == len(scores)
    assert not oof.duplicated(["scenario", "cell_line", "pair_id"]).any()
    assert oof.groupby(group_column)["fold"].nunique().eq(1).all()
    assert len(summary["folds"]) == 5
    assert oof["actual"].notna().all()


def _load_scores_from_frame(scores):
    scores = scores.copy()
    scores["pair_id"] = [
        json.dumps((row.drug_a, row.drug_b))
        for row in scores.itertuples(index=False)
    ]
    return scores


def test_training_requires_passing_official_score_validation(tmp_path):
    score_path = tmp_path / "scores.csv"
    manifest_path = tmp_path / "manifest.json"
    _scores().to_csv(score_path, index=False)
    manifest_path.write_text(json.dumps({"official_score_validation": {"validation_passed": False}}))

    with pytest.raises(ValueError, match="passing official-score sample validation"):
        train_almanac_baseline(
            score_path=score_path,
            manifest_path=manifest_path,
            output_dir=tmp_path / "artifacts",
        )


def test_training_writes_oof_predictions_and_exploratory_model_bundle(tmp_path):
    score_path = tmp_path / "scores.csv"
    manifest_path = tmp_path / "manifest.json"
    output_dir = tmp_path / "artifacts"
    _scores().to_csv(score_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "training_ready": False,
                "official_score_validation": {"validation_passed": True},
            }
        )
    )

    outputs = train_almanac_baseline(
        score_path=score_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
    )

    summary = json.loads(outputs["summary"].read_text())
    predictions = pd.read_csv(outputs["predictions"])
    bundle = joblib.load(outputs["model"])
    features = bundle["vectorizer"].transform(
        [
            {
                "drug:Drug 0": 1.0,
                "drug:Drug 1": 1.0,
                "cell_line:Cell 0": 1.0,
            }
        ]
    )
    prediction = bundle["model"].predict(features)
    assert summary["n_rows"] == len(_scores())
    assert summary["training_ready_manifest_value"] is False
    assert summary["scope_warning"].startswith("Exploratory baseline only")
    assert predictions["scenario"].nunique() == 2
    assert outputs["model"].is_file()
    assert np.isfinite(prediction).all()
