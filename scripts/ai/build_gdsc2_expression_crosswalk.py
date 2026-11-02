#!/usr/bin/env python3
"""Build an auditable exact-name crosswalk from E-MTAB-2770 to GDSC2 graph nodes.

Only case-insensitive exact cell-line names are accepted. Ambiguous and unmatched
identities are retained in the output with explicit statuses; aliases are not guessed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPRESSION = ROOT / "data/raw/expression/E-MTAB-2770_tpmss.tsv"
DEFAULT_RESPONSES = ROOT / "data/raw/gdsc/gdsc2_responses.csv"
DEFAULT_GRAPH = ROOT / "data/processed/graph_v2/typed_graph_v2.pt"
DEFAULT_OUTPUT = ROOT / "data/ai/audits/gdsc2_expression_cell_line_crosswalk.csv"
DEFAULT_SUMMARY = ROOT / "data/ai/audits/gdsc2_expression_cell_line_crosswalk_summary.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_expression_sample_labels(path: Path) -> list[str]:
    """Read and validate the TSV header without loading the 56k-row matrix."""
    if not path.is_file():
        raise FileNotFoundError(f"Expression matrix file is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw_line in handle:
            if not raw_line.strip() or raw_line.startswith("#"):
                continue
            header = next(csv.reader([raw_line.rstrip("\r\n")], delimiter="\t"))
            if header[:2] != ["Gene ID", "Gene Name"]:
                raise ValueError("Unexpected Expression Atlas matrix header.")
            labels = [value.strip() for value in header[2:]]
            if len(labels) != 1019:
                raise ValueError(
                    f"Expected 1,019 expression samples, found {len(labels)}."
                )
            if any(not value for value in labels):
                raise ValueError("Expression matrix header contains an empty sample label.")
            return labels
    raise ValueError("Expression matrix contains no header row.")


def _load_graph_cell_line_ids(path: Path) -> set[str]:
    """Read the locally generated PyG graph and return its typed cell-line IDs."""
    if not path.is_file():
        raise FileNotFoundError(f"Typed graph file is missing: {path}")
    # Only load graph artifacts produced by the user's trusted local pipeline.
    import torch

    graph = torch.load(path, map_location="cpu", weights_only=False)
    if "cell_line" not in graph.node_types:
        raise ValueError("Typed graph has no 'cell_line' node store.")
    node_ids = list(graph["cell_line"].node_ids)
    if len(node_ids) != int(graph["cell_line"].num_nodes):
        raise ValueError("Graph cell-line node_ids do not match its node count.")
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Graph contains duplicate cell-line node IDs.")
    return {str(value) for value in node_ids}


def _expression_cell_line_name(sample_label: str) -> str:
    """Match the existing Ridge pipeline: take text before the final ', '."""
    name, separator, _disease_label = sample_label.rpartition(", ")
    return name.strip() if separator else ""


def build_crosswalk(
    responses: pd.DataFrame,
    sample_labels: list[str],
    graph_cell_line_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create one row per GDSC2 Sanger model with explicit match outcomes."""
    required = {"sanger_model_id", "cell_line", "reported_cancer_type"}
    missing = required - set(responses.columns)
    if missing:
        raise ValueError(f"GDSC2 responses are missing columns: {sorted(missing)}")

    frame = responses[list(required)].copy()
    frame["sanger_model_id"] = frame["sanger_model_id"].fillna("").astype(str).str.strip()
    frame["cell_line"] = frame["cell_line"].fillna("").astype(str).str.strip()
    frame["reported_cancer_type"] = (
        frame["reported_cancer_type"].fillna("").astype(str).str.strip()
    )
    if frame[["sanger_model_id", "cell_line"]].eq("").any().any():
        raise ValueError("GDSC2 response rows contain empty Sanger IDs or cell-line names.")

    names_per_model = frame.groupby("sanger_model_id")["cell_line"].nunique()
    conflicting_models = names_per_model[names_per_model != 1]
    if not conflicting_models.empty:
        raise ValueError(
            "A Sanger model ID maps to multiple GDSC2 cell-line names: "
            f"{conflicting_models.index.tolist()[:20]}"
        )

    def join_cancer_types(values: pd.Series) -> str:
        unique_values = sorted({str(value).strip() for value in values if str(value).strip()})
        return ";".join(unique_values)

    models = (
        frame.groupby("sanger_model_id", sort=True, as_index=False)
        .agg(
            gdsc_cell_line=("cell_line", "first"),
            reported_cancer_types=("reported_cancer_type", join_cancer_types),
            gdsc_response_rows=("sanger_model_id", "size"),
        )
    )

    expression_columns: dict[str, list[int]] = defaultdict(list)
    expression_labels_by_key: dict[str, list[str]] = defaultdict(list)
    for index, label in enumerate(sample_labels):
        name = _expression_cell_line_name(label)
        if name:
            key = name.casefold()
            expression_columns[key].append(index)
            expression_labels_by_key[key].append(label)

    response_model_ids_by_key: dict[str, set[str]] = defaultdict(set)
    for row in models.itertuples(index=False):
        response_model_ids_by_key[row.gdsc_cell_line.casefold()].add(
            row.sanger_model_id
        )

    graph_ids = {str(value) for value in graph_cell_line_ids}
    output_rows: list[dict[str, Any]] = []
    for row in models.itertuples(index=False):
        name_key = row.gdsc_cell_line.casefold()
        candidate_columns = expression_columns.get(name_key, [])
        candidate_model_ids = response_model_ids_by_key[name_key]
        graph_node_id = f"GDSC_MODEL:{row.sanger_model_id}"

        if len(candidate_model_ids) > 1 or len(candidate_columns) > 1:
            match_status = "ambiguous"
            matched_index = None
            matched_label = ""
        elif len(candidate_columns) == 1:
            match_status = "matched_exact_casefold"
            matched_index = candidate_columns[0]
            matched_label = sample_labels[matched_index]
        else:
            match_status = "unmatched"
            matched_index = None
            matched_label = ""

        output_rows.append(
            {
                "sanger_model_id": row.sanger_model_id,
                "gdsc_cell_line": row.gdsc_cell_line,
                "reported_cancer_types": row.reported_cancer_types,
                "gdsc_response_rows": int(row.gdsc_response_rows),
                "graph_node_id": graph_node_id,
                "graph_node_present": graph_node_id in graph_ids,
                "expression_sample_column_index_0based": matched_index,
                "expression_sample_label": matched_label,
                "expression_cell_line_name": (
                    _expression_cell_line_name(matched_label) if matched_label else ""
                ),
                "expression_candidate_count": len(candidate_columns),
                "gdsc_model_name_candidate_count": len(candidate_model_ids),
                "match_status": match_status,
                "match_rule": "case_insensitive_exact_cell_line_name",
            }
        )

    crosswalk = pd.DataFrame(output_rows)
    if crosswalk["sanger_model_id"].duplicated().any():
        raise ValueError("Crosswalk unexpectedly contains duplicate Sanger model IDs.")

    status_counts = crosswalk["match_status"].value_counts().to_dict()
    matched = crosswalk[crosswalk["match_status"] == "matched_exact_casefold"]
    graph_present = int(crosswalk["graph_node_present"].sum())
    expression_indices = matched["expression_sample_column_index_0based"].dropna()
    if expression_indices.duplicated().any():
        raise ValueError(
            "Two GDSC2 models map to the same expression matrix column."
        )

    summary: dict[str, Any] = {
        "status": "passed" if graph_present == len(crosswalk) else "warning",
        "matching_policy": (
            "case-insensitive exact cell-line name only; no aliases, punctuation "
            "normalisation, disease-label matching, or fuzzy matching"
        ),
        "expression_label_parser": "text before the final ', ' separator",
        "graph_node_id_rule": "GDSC_MODEL:<sanger_model_id>",
        "response_models": int(len(crosswalk)),
        "response_rows": int(frame.shape[0]),
        "expression_samples": int(len(sample_labels)),
        "matched_models": int(len(matched)),
        "ambiguous_models": int(status_counts.get("ambiguous", 0)),
        "unmatched_models": int(status_counts.get("unmatched", 0)),
        "graph_cell_line_nodes_expected": int(len(crosswalk)),
        "graph_cell_line_nodes_present": graph_present,
        "graph_cell_line_nodes_missing": int(len(crosswalk) - graph_present),
        "matched_models_with_graph_nodes": int(matched["graph_node_present"].sum()),
        "unique_expression_columns_used": int(expression_indices.nunique()),
        "duplicate_expression_column_assignments": 0,
        "match_status_counts": {
            "matched_exact_casefold": int(status_counts.get("matched_exact_casefold", 0)),
            "ambiguous": int(status_counts.get("ambiguous", 0)),
            "unmatched": int(status_counts.get("unmatched", 0)),
        },
        "graph_missing_sanger_model_ids": crosswalk.loc[
            ~crosswalk["graph_node_present"], "sanger_model_id"
        ].tolist(),
    }
    return crosswalk, summary


def main(
    expression_file: Path = DEFAULT_EXPRESSION,
    response_file: Path = DEFAULT_RESPONSES,
    graph_file: Path = DEFAULT_GRAPH,
    output_file: Path = DEFAULT_OUTPUT,
    summary_file: Path = DEFAULT_SUMMARY,
) -> dict[str, Any]:
    expression_file = Path(expression_file).expanduser()
    response_file = Path(response_file).expanduser()
    graph_file = Path(graph_file).expanduser()
    output_file = Path(output_file).expanduser()
    summary_file = Path(summary_file).expanduser()

    for path, label in (
        (response_file, "GDSC2 response file"),
        (graph_file, "typed graph file"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")

    sample_labels = _read_expression_sample_labels(expression_file)
    responses = pd.read_csv(
        response_file,
        dtype={"sanger_model_id": str, "cell_line": str, "reported_cancer_type": str},
    )
    graph_cell_line_ids = _load_graph_cell_line_ids(graph_file)
    crosswalk, summary = build_crosswalk(
        responses, sample_labels, graph_cell_line_ids
    )

    summary["inputs"] = {
        "expression_file": str(expression_file),
        "expression_file_sha256": _sha256_file(expression_file),
        "response_file": str(response_file),
        "response_file_sha256": _sha256_file(response_file),
        "graph_file": str(graph_file),
        "graph_cell_line_id_count": len(graph_cell_line_ids),
    }
    summary["outputs"] = {
        "crosswalk_csv": str(output_file),
        "summary_json": str(summary_file),
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    crosswalk.to_csv(output_file, index=False)
    summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression-file", type=Path, default=DEFAULT_EXPRESSION)
    parser.add_argument("--response-file", type=Path, default=DEFAULT_RESPONSES)
    parser.add_argument("--graph-file", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    main(
        expression_file=args.expression_file,
        response_file=args.response_file,
        graph_file=args.graph_file,
        output_file=args.output_file,
        summary_file=args.summary_file,
    )
