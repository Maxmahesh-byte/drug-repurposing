from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
SCORE_SOURCE = ROOT / "data/processed/almanac/nci_almanac_combo_scores.csv"
MANIFEST_SOURCE = ROOT / "data/processed/almanac/nci_almanac_manifest.json"
OUTPUT_DIR = ROOT / "artifacts/almanac_baseline"
N_SPLITS = 5
RIDGE_ALPHA = 100.0
REQUIRED_COLUMNS = {
    "cell_line",
    "drug_a",
    "drug_b",
    "combo_score",
    "score_complete",
}


def _pair_id(drug_a: str, drug_b: str) -> str:
    return json.dumps((drug_a, drug_b), ensure_ascii=True)


def _load_scores(path: Path = SCORE_SOURCE) -> pd.DataFrame:
    scores = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(scores.columns)
    if missing_columns:
        raise ValueError(f"ALMANAC score table is missing columns: {sorted(missing_columns)}")
    scores = scores.loc[
        scores["score_complete"].astype("string").str.casefold().eq("true")
        & scores["combo_score"].notna()
    ].copy()
    if not np.isfinite(scores["combo_score"].to_numpy(dtype=float)).all():
        raise ValueError("ALMANAC ComboScores must be finite")
    if scores[["cell_line", "drug_a", "drug_b"]].isna().any().any():
        raise ValueError("ALMANAC training rows must have cell-line and drug identities")
    scores["pair_id"] = [
        _pair_id(drug_a, drug_b)
        for drug_a, drug_b in scores[["drug_a", "drug_b"]].itertuples(
            index=False, name=None
        )
    ]
    if scores.duplicated(["cell_line", "pair_id"]).any():
        raise ValueError("ALMANAC data contain duplicate cell-line/drug-pair labels")
    if scores["pair_id"].nunique() < N_SPLITS:
        raise ValueError(f"At least {N_SPLITS} distinct drug pairs are required")
    if scores["cell_line"].nunique() < N_SPLITS:
        raise ValueError(f"At least {N_SPLITS} distinct cell lines are required")
    return scores.reset_index(drop=True)


def _feature_records(
    scores: pd.DataFrame,
    include_cell_line: bool,
) -> list[dict[str, float]]:
    features = []
    for row in scores.itertuples(index=False):
        record = {
            f"drug:{row.drug_a}": 1.0,
            f"drug:{row.drug_b}": 1.0,
        }
        if include_cell_line:
            record[f"cell_line:{row.cell_line}"] = 1.0
        features.append(record)
    return features


def _fit_ridge(
    train: pd.DataFrame,
    test: pd.DataFrame,
    include_cell_line: bool,
    alpha: float = RIDGE_ALPHA,
) -> np.ndarray:
    vectorizer = DictVectorizer(sparse=True)
    train_features = vectorizer.fit_transform(
        _feature_records(train, include_cell_line=include_cell_line)
    )
    test_features = vectorizer.transform(
        _feature_records(test, include_cell_line=include_cell_line)
    )
    model = Ridge(alpha=alpha, solver="lsqr")
    model.fit(train_features, train["combo_score"].to_numpy(dtype=float))
    return model.predict(test_features)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    residuals = actual - predicted
    total_sum_squares = float(np.square(actual - actual.mean()).sum())
    return {
        "n": int(len(actual)),
        "mae": float(np.abs(residuals).mean()),
        "rmse": float(np.sqrt(np.square(residuals).mean())),
        "r2": (
            float(1 - np.square(residuals).sum() / total_sum_squares)
            if total_sum_squares > 0
            else None
        ),
    }


def _grouped_oof(
    scores: pd.DataFrame,
    group_column: str,
    scenario: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    splitter = GroupKFold(n_splits=N_SPLITS)
    groups = scores[group_column].to_numpy()
    prediction_frames = []
    fold_summaries = []

    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(scores, groups=groups),
        start=1,
    ):
        train = scores.iloc[train_indices]
        test = scores.iloc[test_indices]
        train_mean = float(train["combo_score"].mean())
        actual = test["combo_score"].to_numpy(dtype=float)

        predictions: dict[str, np.ndarray] = {
            "global_mean": np.full(len(test), train_mean, dtype=float)
        }
        if scenario == "drug_pair_holdout":
            line_means = train.groupby("cell_line")["combo_score"].mean()
            predictions["cell_line_mean"] = (
                test["cell_line"].map(line_means).fillna(train_mean).to_numpy(dtype=float)
            )
            predictions["drug_cell_additive_ridge"] = _fit_ridge(
                train,
                test,
                include_cell_line=True,
            )
        elif scenario == "cell_line_holdout":
            pair_means = train.groupby("pair_id")["combo_score"].mean()
            predictions["drug_pair_mean"] = (
                test["pair_id"].map(pair_means).fillna(train_mean).to_numpy(dtype=float)
            )
            predictions["drug_additive_ridge"] = _fit_ridge(
                train,
                test,
                include_cell_line=False,
            )
        else:
            raise ValueError(f"Unsupported ALMANAC validation scenario: {scenario}")

        prediction_frame = test[
            ["cell_line", "drug_a", "drug_b", "pair_id", "combo_score"]
        ].copy()
        prediction_frame = prediction_frame.rename(columns={"combo_score": "actual"})
        prediction_frame.insert(0, "scenario", scenario)
        prediction_frame.insert(1, "fold", fold)
        for model_name, values in predictions.items():
            prediction_frame[model_name] = values
        prediction_frames.append(prediction_frame)

        fold_summaries.append(
            {
                "fold": fold,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_groups": int(train[group_column].nunique()),
                "test_groups": int(test[group_column].nunique()),
                "metrics": {
                    model_name: _metrics(actual, values)
                    for model_name, values in predictions.items()
                },
            }
        )

    oof = pd.concat(prediction_frames, ignore_index=True)
    if len(oof) != len(scores) or oof.duplicated(
        ["scenario", "cell_line", "pair_id"]
    ).any():
        raise RuntimeError(f"{scenario} cross-validation did not predict each row once")

    model_names = [
        column
        for column in oof.columns
        if column
        in {
            "global_mean",
            "cell_line_mean",
            "drug_cell_additive_ridge",
            "drug_pair_mean",
            "drug_additive_ridge",
        }
    ]
    summary = {
        "grouped_by": group_column,
        "folds": fold_summaries,
        "overall_metrics": {
            model_name: _metrics(
                oof["actual"].to_numpy(dtype=float),
                oof[model_name].to_numpy(dtype=float),
            )
            for model_name in model_names
        },
    }
    return oof, summary


def _fit_final_model(scores: pd.DataFrame, path: Path) -> None:
    vectorizer = DictVectorizer(sparse=True)
    features = vectorizer.fit_transform(
        _feature_records(scores, include_cell_line=True)
    )
    model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr")
    model.fit(features, scores["combo_score"].to_numpy(dtype=float))
    joblib.dump(
        {
            "schema_version": 1,
            "target": "NCI-ALMANAC normalized ComboScore",
            "feature_definition": (
                "Additive one-hot drug identities plus cell-line identity; no "
                "molecular, pathway, or pediatric-specific features."
            ),
            "ridge_alpha": RIDGE_ALPHA,
            "vectorizer": vectorizer,
            "model": model,
        },
        path,
    )


def train_almanac_baseline(
    score_path: Path = SCORE_SOURCE,
    manifest_path: Path = MANIFEST_SOURCE,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Path]:
    scores = _load_scores(score_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"ALMANAC manifest is required: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    validation = manifest.get("official_score_validation", {})
    if not validation.get("validation_passed", False):
        raise ValueError(
            "ALMANAC baseline requires a passing official-score sample validation"
        )

    pair_oof, pair_summary = _grouped_oof(
        scores,
        group_column="pair_id",
        scenario="drug_pair_holdout",
    )
    cell_oof, cell_summary = _grouped_oof(
        scores,
        group_column="cell_line",
        scenario="cell_line_holdout",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "oof_predictions.csv"
    summary_path = output_dir / "metrics.json"
    model_path = output_dir / "almanac_additive_ridge.joblib"

    pd.concat([pair_oof, cell_oof], ignore_index=True).to_csv(
        prediction_path,
        index=False,
    )
    _fit_final_model(scores, model_path)

    summary = {
        "task": "General cancer-cell-line NCI-ALMANAC combination ComboScore regression",
        "scope_warning": (
            "Exploratory baseline only; ALMANAC is not pediatric-specific and this "
            "model does not establish pediatric-cancer performance."
        ),
        "label": "NCI-ALMANAC normalized ComboScore",
        "source": str(score_path),
        "training_ready_manifest_value": bool(manifest.get("training_ready", False)),
        "official_score_sample_validation_passed": bool(
            validation["validation_passed"]
        ),
        "n_rows": int(len(scores)),
        "n_drug_pairs": int(scores["pair_id"].nunique()),
        "n_cell_lines": int(scores["cell_line"].nunique()),
        "n_drugs": int(pd.unique(scores[["drug_a", "drug_b"]].to_numpy().ravel()).size),
        "label_summary": {
            "mean": float(scores["combo_score"].mean()),
            "std": float(scores["combo_score"].std()),
            "median": float(scores["combo_score"].median()),
            "min": float(scores["combo_score"].min()),
            "max": float(scores["combo_score"].max()),
        },
        "cross_validation": {
            "n_splits": N_SPLITS,
            "split_policy": (
                "Deterministic, unshuffled GroupKFold; all observations for a drug pair are "
                "held out together in drug_pair_holdout, and all observations for a "
                "cell line are held out together in cell_line_holdout."
            ),
            "ridge_alpha": RIDGE_ALPHA,
            "drug_pair_holdout": pair_summary,
            "cell_line_holdout": cell_summary,
        },
        "final_model": {
            "estimator": "scikit-learn Ridge regression",
            "features": "drug identity main effects and cell-line identity main effect",
            "limitations": [
                "No molecular or pathway features.",
                "No drug-pair interaction features.",
                "No pediatric-specific training or validation data.",
                "Does not predict dose-level response or clinical efficacy.",
            ],
        },
        "outputs": {
            "predictions": str(prediction_path),
            "model": str(model_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return {
        "summary": summary_path,
        "predictions": prediction_path,
        "model": model_path,
    }


def main() -> None:
    outputs = train_almanac_baseline()
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
