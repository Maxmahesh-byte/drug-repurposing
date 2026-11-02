#!/usr/bin/env python3
"""Prepare a leakage-aware GDSC2 response-pair dataset for graph experiments.

The output keeps measured response labels in a standalone pair table. It does
not insert LN(IC50) or AUC into graph node/edge features and does not modify
the immutable graph_v2/graph_v3 artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ai.train_gdsc2_expression_model import (
    FOLD_SOURCE,
    RESPONSE_SOURCE,
    _read_expression_matrix,
    _read_responses,
)

DEFAULT_EXPRESSION = ROOT / "data/raw/expression/E-MTAB-2770_tpmss.tsv"
DEFAULT_CROSSWALK = ROOT / "data/ai/audits/gdsc2_expression_cell_line_crosswalk.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data/ai/response_dataset"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expression_name(sample_label: str) -> str:
    name, separator, _diagnosis = str(sample_label).rpartition(", ")
    return name.strip() if separator else ""


def assemble_response_dataset(
    responses: pd.DataFrame,
    crosswalk: pd.DataFrame,
    sample_labels: list[str],
    gene_ids: list[str],
    expression_log2_tpm: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[int]]:
    """Join matched pairs and return model-ordered expression rows.

    expression_log2_tpm has shape (genes, expression samples) and is already
    log2(TPM + 1), matching the Ridge benchmark's input transform. Z-scoring
    and feature selection are intentionally deferred to each training fold.
    """
    required_response = {
        "curve_id", "drug_id", "drug_name", "cell_line", "sanger_model_id",
        "reported_cancer_type", "fold", "ln_ic50", "auc",
    }
    required_crosswalk = {
        "sanger_model_id", "gdsc_cell_line", "graph_node_id",
        "graph_node_present", "expression_sample_column_index_0based",
        "expression_sample_label", "match_status",
    }
    missing_response = required_response - set(responses.columns)
    missing_crosswalk = required_crosswalk - set(crosswalk.columns)
    if missing_response:
        raise ValueError(f"Response table missing columns: {sorted(missing_response)}")
    if missing_crosswalk:
        raise ValueError(f"Crosswalk missing columns: {sorted(missing_crosswalk)}")

    if expression_log2_tpm.ndim != 2:
        raise ValueError("Expression matrix must be two-dimensional (genes x samples).")
    if expression_log2_tpm.shape != (len(gene_ids), len(sample_labels)):
        raise ValueError(
            "Expression matrix dimensions do not match gene IDs and sample labels."
        )
    if not np.isfinite(expression_log2_tpm).all():
        raise ValueError("Expression matrix contains non-finite values.")
    if len(gene_ids) != len(set(gene_ids)):
        raise ValueError("Expression gene IDs are not unique.")

    cw = crosswalk.copy()
    cw["sanger_model_id"] = cw["sanger_model_id"].astype(str).str.strip()
    if cw["sanger_model_id"].eq("").any() or cw["sanger_model_id"].duplicated().any():
        raise ValueError("Crosswalk must contain one nonblank row per Sanger model ID.")

    matched = cw[cw["match_status"] == "matched_exact_casefold"].copy()
    if matched.empty:
        raise ValueError("Crosswalk contains no exact-name matched models.")
    graph_present = matched["graph_node_present"].map(
        lambda value: str(value).strip().casefold() in {"true", "1"}
    )
    if not graph_present.all():
        raise ValueError("At least one matched model is missing its typed graph node.")

    matched["expression_sample_column_index_0based"] = pd.to_numeric(
        matched["expression_sample_column_index_0based"], errors="raise"
    ).astype(int)
    columns = matched["expression_sample_column_index_0based"].tolist()
    if any(index < 0 or index >= len(sample_labels) for index in columns):
        raise ValueError("Crosswalk contains an out-of-range expression column index.")
    if len(columns) != len(set(columns)):
        raise ValueError("Crosswalk assigns one expression column to multiple models.")

    for row in matched.itertuples(index=False):
        column = int(row.expression_sample_column_index_0based)
        observed_label = sample_labels[column]
        if str(row.expression_sample_label) != observed_label:
            raise ValueError(
                f"Crosswalk sample label changed at expression column {column}."
            )
        if expression_name(observed_label).casefold() != str(
            row.gdsc_cell_line
        ).strip().casefold():
            raise ValueError(
                f"Crosswalk exact-name identity check failed for {row.sanger_model_id}."
            )
        expected_node = f"GDSC_MODEL:{row.sanger_model_id}"
        if str(row.graph_node_id) != expected_node:
            raise ValueError(
                f"Unexpected graph node ID for Sanger model {row.sanger_model_id}."
            )

    matched = matched.sort_values("sanger_model_id").reset_index(drop=True)
    model_ids = matched["sanger_model_id"].astype(str).tolist()
    expression_columns = matched[
        "expression_sample_column_index_0based"
    ].to_numpy(dtype=np.int64)
    model_expression = expression_log2_tpm[:, expression_columns].T.copy()
    model_to_expression_row = {
        model_id: index for index, model_id in enumerate(model_ids)
    }

    source = responses.copy()
    source["sanger_model_id"] = source["sanger_model_id"].astype(str).str.strip()
    source["curve_id"] = source["curve_id"].astype(str).str.strip()
    source["drug_id"] = source["drug_id"].astype(str).str.strip()
    if source["curve_id"].eq("").any() or source["curve_id"].duplicated().any():
        raise ValueError("Response table has blank or duplicate curve IDs.")

    selected = source[source["sanger_model_id"].isin(model_to_expression_row)].copy()
    if selected.empty:
        raise ValueError("No response measurements match exact-name crosswalk models.")
    selected = selected.merge(
        matched[["sanger_model_id", "gdsc_cell_line", "graph_node_id"]],
        on="sanger_model_id",
        how="left",
        validate="many_to_one",
    )
    if not selected["cell_line"].astype(str).str.casefold().eq(
        selected["gdsc_cell_line"].astype(str).str.casefold()
    ).all():
        raise ValueError("Response cell-line names disagree with the audited crosswalk.")

    selected["expression_row_index"] = selected["sanger_model_id"].map(
        model_to_expression_row
    ).astype(int)
    selected["expression_column_index_0based"] = selected["sanger_model_id"].map(
        matched.set_index("sanger_model_id")[
            "expression_sample_column_index_0based"
        ]
    ).astype(int)

    numeric_folds = pd.to_numeric(selected["fold"], errors="coerce")
    if numeric_folds.isna().any() or not np.equal(
        numeric_folds.to_numpy(dtype=float),
        np.floor(numeric_folds.to_numpy(dtype=float)),
    ).all():
        raise ValueError("Some response rows have missing or non-integer frozen fold IDs.")
    if not numeric_folds.isin(range(5)).all():
        invalid_folds = sorted(numeric_folds[~numeric_folds.isin(range(5))].unique().tolist())
        raise ValueError(f"Unexpected frozen fold IDs outside 0-4: {invalid_folds}.")
    selected["fold"] = numeric_folds.astype(int)
    folds_per_model = selected.groupby("sanger_model_id")["fold"].nunique()
    if folds_per_model.gt(1).any():
        raise ValueError("A Sanger model ID occurs in more than one held-out fold.")
    for target in ("ln_ic50", "auc"):
        values = pd.to_numeric(selected[target], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Target {target} contains missing or non-finite values.")

    if selected["curve_id"].duplicated().any():
        raise ValueError("Selected response pairs contain duplicate curve IDs.")
    selected = selected.sort_values(
        ["fold", "sanger_model_id", "drug_id", "curve_id"]
    ).reset_index(drop=True)

    columns_out = [
        "curve_id", "drug_id", "drug_name", "cell_line", "sanger_model_id",
        "graph_node_id", "reported_cancer_type", "fold",
        "expression_row_index", "expression_column_index_0based",
        "ln_ic50", "auc",
    ]
    return selected[columns_out], model_expression, model_ids, expression_columns.tolist()


def main(
    expression_file: Path = DEFAULT_EXPRESSION,
    crosswalk_file: Path = DEFAULT_CROSSWALK,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    expression_file = Path(expression_file).expanduser()
    crosswalk_file = Path(crosswalk_file).expanduser()
    output_dir = Path(output_dir).expanduser()
    for path, label in (
        (expression_file, "expression matrix"),
        (crosswalk_file, "audited expression crosswalk"),
        (RESPONSE_SOURCE, "GDSC2 response source"),
        (FOLD_SOURCE, "frozen GDSC2 fold assignments"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required {label} is missing: {path}")

    print("Loading and validating the frozen GDSC2 response/fold table...", flush=True)
    responses = _read_responses()
    crosswalk = pd.read_csv(
        crosswalk_file,
        dtype={"sanger_model_id": str, "graph_node_id": str},
    )
    print("Reading E-MTAB-2770 and applying log2(TPM + 1)...", flush=True)
    sample_labels, gene_ids, expression, expression_digest = _read_expression_matrix(
        expression_file
    )
    pairs, model_expression, model_ids, expression_columns = assemble_response_dataset(
        responses, crosswalk, sample_labels, gene_ids, expression
    )
    found_folds = sorted(pairs["fold"].unique().tolist())
    if found_folds != list(range(5)):
        raise ValueError(
            "The complete matched real-data benchmark must contain all five "
            f"frozen folds 0-4; found {found_folds}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = output_dir / "gdsc2_matched_response_pairs.csv"
    expression_path = output_dir / "gdsc2_matched_expression_features.npz"
    summary_path = output_dir / "gdsc2_matched_response_dataset_summary.json"

    pairs.to_csv(pair_path, index=False)
    np.savez_compressed(
        expression_path,
        expression_log2_tpm=model_expression.astype(np.float32),
        gene_ids=np.asarray(gene_ids, dtype=str),
        sanger_model_ids=np.asarray(model_ids, dtype=str),
        expression_column_indices_0based=np.asarray(expression_columns, dtype=np.int64),
    )

    summary: dict[str, Any] = {
        "status": "prepared",
        "scope": (
            "GDSC2 in-vitro single-agent response measurements for exact-name "
            "E-MTAB-2770-matched cell lines; known drugs and held-out cell lines"
        ),
        "matched_models": int(len(model_ids)),
        "response_rows": int(len(pairs)),
        "drugs": int(pairs["drug_id"].nunique()),
        "genes": int(len(gene_ids)),
        "folds": {
            str(int(fold)): {
                "models": int(group["sanger_model_id"].nunique()),
                "rows": int(len(group)),
            }
            for fold, group in pairs.groupby("fold", sort=True)
        },
        "validation": {
            "five_folds_present": True,
            "sanger_model_ids_confined_to_one_fold": True,
            "exact_name_crosswalk_only": True,
            "matched_models_have_graph_nodes": True,
            "duplicate_expression_columns": 0,
            "response_labels_in_expression_features": False,
            "response_edges_required_for_supervision": False,
            "response_edge_attributes_permitted_as_input": False,
            "expression_transform": "log2(TPM + 1); fold-specific z-score/feature selection not yet applied",
        },
        "inputs": {
            "expression_file": str(expression_file),
            "expression_file_sha256": sha256_file(expression_file),
            "expression_parsed_content_sha256": expression_digest,
            "crosswalk_file": str(crosswalk_file),
            "crosswalk_file_sha256": sha256_file(crosswalk_file),
            "response_file": str(RESPONSE_SOURCE),
            "response_file_sha256": sha256_file(RESPONSE_SOURCE),
            "fold_file": str(FOLD_SOURCE),
            "fold_file_sha256": sha256_file(FOLD_SOURCE),
        },
        "outputs": {
            "response_pairs_csv": str(pair_path),
            "expression_features_npz": str(expression_path),
            "summary_json": str(summary_path),
        },
        "important_limitations": [
            "This is a data-preparation artifact, not a trained GNN.",
            "LN_IC50 and AUC are labels only and must never enter message-passing inputs.",
            "The GDSC2 gdsc2_response relation must be excluded from model inputs, or filtered strictly to training-fold edges before every fold.",
            "Standardisation and expression feature selection must be fitted on training-fold cell lines only.",
            "The unmatched GDSC2 cell lines are intentionally excluded; no expression values are imputed.",
            "These are cell-line measurements, not patient outcomes or evidence of clinical efficacy.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression-file", type=Path, default=DEFAULT_EXPRESSION)
    parser.add_argument("--crosswalk-file", type=Path, default=DEFAULT_CROSSWALK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    main(args.expression_file, args.crosswalk_file, args.output_dir)
