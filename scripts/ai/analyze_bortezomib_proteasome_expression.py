#!/usr/bin/env python3
"""Test a fixed proteasome-complex expression score against GDSC2 bortezomib response."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
from statsmodels.formula.api import ols

ROOT = Path(__file__).resolve().parents[2]
DATABASE = ROOT / "data/processed/graph_v3/research_evidence.sqlite"
OUTPUT_DIR = ROOT / "reports"
MODEL_OUTPUT = OUTPUT_DIR / "bortezomib_proteasome_expression_models.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "bortezomib_proteasome_expression_summary.json"
EXPRESSION_URL = (
    "https://www.ebi.ac.uk/gxa/experiments-content/E-MTAB-2770/resources/"
    "ExperimentDownloadSupplier.RnaSeqBaseline/tpmss.tsv"
)

# Fixed before evaluating any expression-response associations.
PROTEASOME_GENES = (
    "ADRM1",
    "PSMA1",
    "PSMA2",
    "PSMA3",
    "PSMA4",
    "PSMA5",
    "PSMA6",
    "PSMA7",
    "PSMA8",
    "PSMB1",
    "PSMB2",
    "PSMB3",
    "PSMB4",
    "PSMB5",
    "PSMB6",
    "PSMB7",
    "PSMB8",
    "PSMB9",
    "PSMB10",
    "PSMB11",
    "PSMC1",
    "PSMC2",
    "PSMC3",
    "PSMC4",
    "PSMC5",
    "PSMC6",
    "PSMD1",
    "PSMD2",
    "PSMD3",
    "PSMD4",
    "PSMD6",
    "PSMD7",
    "PSMD8",
    "PSMD11",
    "PSMD12",
    "PSMD13",
    "PSMD14",
    "SEM1",
)


def _read_expression_rows() -> tuple[list[str], dict[str, list[float]], str]:
    request = requests.get(
        EXPRESSION_URL,
        headers={"User-Agent": "drug-repurposing-research/0.1"},
        stream=True,
        timeout=(15, 120),
    )
    request.raise_for_status()

    sample_names: list[str] | None = None
    expression: dict[str, list[float]] = {}
    digest = hashlib.sha256()
    wanted = set(PROTEASOME_GENES)

    try:
        for raw_line in request.iter_lines(chunk_size=1024 * 1024):
            if not raw_line or raw_line.startswith(b"#"):
                continue
            row = next(csv.reader([raw_line.decode("utf-8")], delimiter="\t"))
            if sample_names is None:
                if row[:2] != ["Gene ID", "Gene Name"]:
                    raise ValueError("Unexpected Expression Atlas matrix header.")
                sample_names = row[2:]
                if len(sample_names) != 1019:
                    raise ValueError(
                        f"Expected 1,019 expression samples, found {len(sample_names)}."
                    )
                continue

            gene = row[1].strip().upper()
            if gene not in wanted:
                continue
            if gene in expression:
                raise ValueError(f"Expression matrix contains duplicate gene row: {gene}")
            if len(row) != len(sample_names) + 2:
                raise ValueError(f"Unexpected sample count for expression gene {gene}.")

            digest.update(raw_line)
            digest.update(b"\n")
            try:
                # Expression Atlas leaves unexpressed TPM entries blank.
                values = [float(value) if value else 0.0 for value in row[2:]]
            except ValueError as error:
                raise ValueError(f"Non-numeric TPM value for gene {gene}.") from error
            if not np.isfinite(values).all() or any(value < 0 for value in values):
                raise ValueError(f"Invalid TPM values for gene {gene}.")
            expression[gene] = values

            if len(expression) == len(PROTEASOME_GENES):
                break
    finally:
        request.close()

    missing = sorted(wanted - expression.keys())
    if missing:
        raise ValueError(f"Expression matrix is missing component genes: {missing}")
    if sample_names is None:
        raise ValueError("Expression Atlas response did not contain a matrix header.")
    return sample_names, expression, digest.hexdigest()


def _load_bortezomib_responses() -> pd.DataFrame:
    if not DATABASE.is_file():
        raise FileNotFoundError(f"Research evidence database is missing: {DATABASE}")
    connection = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True)
    try:
        frame = pd.read_sql_query(
            """
            SELECT reported_cancer_type, cell_line, sanger_model_id,
                   ln_ic50, auc, rmse
            FROM gdsc_response
            WHERE drug_id = '1191'
              AND LOWER(TRIM(drug_name)) = 'bortezomib'
              AND ln_ic50 IS NOT NULL
              AND auc IS NOT NULL
            """,
            connection,
        )
    finally:
        connection.close()

    if frame.empty:
        raise ValueError("No Bortezomib GDSC2 responses were found for drug ID 1191.")
    if frame["sanger_model_id"].duplicated().any():
        raise ValueError("Bortezomib response table has duplicate Sanger model IDs.")
    return frame


def _cell_line_name(label: str) -> str:
    name, separator, _ = label.rpartition(", ")
    return name.strip() if separator else label.strip()


def _fit_adjusted_model(
    frame: pd.DataFrame, outcome: str
) -> dict[str, float | int | None]:
    model = ols(
        f"{outcome} ~ proteasome_expression_score + C(reported_cancer_type)",
        data=frame,
    ).fit(cov_type="HC3")
    term = "proteasome_expression_score"
    return {
        "n_models": int(model.nobs),
        "coefficient_per_score_unit": float(model.params[term]),
        "hc3_standard_error": float(model.bse[term]),
        "ci_95_low": float(model.conf_int().loc[term, 0]),
        "ci_95_high": float(model.conf_int().loc[term, 1]),
        "two_sided_p_value": float(model.pvalues[term]),
        "r_squared": float(model.rsquared),
        "cancer_type_fixed_effect_count": int(frame["reported_cancer_type"].nunique()),
    }


def _write_json(path: Path, payload: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def main() -> None:
    sample_labels, tpm_by_gene, expression_rows_sha256 = _read_expression_rows()
    sample_names = [_cell_line_name(label) for label in sample_labels]
    normalized_names = [name.casefold() for name in sample_names]
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("Expression sample names are not unique after label parsing.")

    expression_frame = pd.DataFrame(tpm_by_gene, index=sample_names)
    log_expression = np.log2(expression_frame + 1.0)
    gene_standard_deviation = log_expression.std(axis=0, ddof=0)
    if (gene_standard_deviation == 0).any():
        constant_genes = gene_standard_deviation[gene_standard_deviation == 0].index.tolist()
        raise ValueError(f"Cannot standardize constant-expression genes: {constant_genes}")
    gene_z_scores = (log_expression - log_expression.mean(axis=0)) / gene_standard_deviation
    expression_frame["proteasome_expression_score"] = gene_z_scores.mean(axis=1)
    expression_frame.index = expression_frame.index.str.casefold()

    responses = _load_bortezomib_responses()
    responses["cell_line_key"] = responses["cell_line"].astype(str).str.strip().str.casefold()
    matched = responses.merge(
        expression_frame[["proteasome_expression_score"]],
        left_on="cell_line_key",
        right_index=True,
        how="inner",
        validate="one_to_one",
    )
    if matched.empty:
        raise ValueError("No exact GDSC-to-expression cell-line name matches were found.")

    all_models = _fit_adjusted_model(matched, "ln_ic50")
    all_models_auc = _fit_adjusted_model(matched, "auc")
    nb = matched[
        matched["reported_cancer_type"].astype(str).str.strip().str.upper() == "NB"
    ].copy()
    nb_spearman_ln = spearmanr(nb["proteasome_expression_score"], nb["ln_ic50"])
    nb_spearman_auc = spearmanr(nb["proteasome_expression_score"], nb["auc"])

    matched["match_method"] = "case_insensitive_exact_cell_line_name"
    matched["expression_source"] = "E-MTAB-2770 RNA-seq TPM"
    matched.sort_values(["reported_cancer_type", "cell_line"], inplace=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matched[
        [
            "reported_cancer_type",
            "cell_line",
            "sanger_model_id",
            "ln_ic50",
            "auc",
            "rmse",
            "proteasome_expression_score",
            "match_method",
            "expression_source",
        ]
    ].to_csv(MODEL_OUTPUT, index=False)

    p_values = sorted([all_models["two_sided_p_value"], all_models_auc["two_sided_p_value"]])
    adjusted_p_values = [
        min(1.0, p_values[0] * 2),
        min(1.0, p_values[1] * 2),
    ]
    summary = {
        "analysis": "Pre-specified mean z-score of 38 proteasome-complex component genes",
        "accessed_utc": datetime.now(timezone.utc).isoformat(),
        "expression_source": {
            "study": "E-MTAB-2770",
            "title": "RNA-seq of 1019 human cancer cell lines from CCLE",
            "download_url": EXPRESSION_URL,
            "scale": "TPM",
            "transform": "log2(TPM + 1), then gene-wise z-score across all 1,019 cell lines",
            "score": "arithmetic mean of the 38 gene-wise z-scores",
            "blank_tpm_policy": "blank TPM entries treated as zero",
            "selected_gene_rows_sha256": expression_rows_sha256,
        },
        "response_source": {
            "study": "GDSC2 fitted dose-response release 2023-10-27",
            "database": str(DATABASE.relative_to(ROOT)),
            "drug_id": "1191",
            "drug_name": "Bortezomib",
            "primary_outcome": "ln_ic50 (two-sided)",
            "secondary_outcome": "AUC (two-sided; Holm-adjusted with primary outcome)",
        },
        "matching": {
            "method": "case-insensitive exact cell-line-name match only; no alias inference",
            "gdsc_response_models": int(len(responses)),
            "expression_models": int(len(sample_names)),
            "matched_models": int(len(matched)),
            "matched_cancer_types": int(matched["reported_cancer_type"].nunique()),
            "matched_neuroblastoma_models": int(len(nb)),
            "all_neuroblastoma_response_models": int(
                (
                    responses["reported_cancer_type"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    == "NB"
                ).sum()
            ),
            "matched_sanger_model_ids": sorted(matched["sanger_model_id"].astype(str).tolist()),
            "identity_limitation": (
                "The expression matrix has cell-line labels, not Sanger model IDs. "
                "Only exact names were joined; this is conservative but is not an "
                "independent stable-ID crosswalk."
            ),
        },
        "results": {
            "primary_lineage_adjusted_ln_ic50": all_models,
            "secondary_lineage_adjusted_auc": all_models_auc,
            "holm_adjusted_p_values": {
                "ln_ic50": float(
                    adjusted_p_values[
                        0 if all_models["two_sided_p_value"] <= all_models_auc["two_sided_p_value"] else 1
                    ]
                ),
                "auc": float(
                    adjusted_p_values[
                        0 if all_models_auc["two_sided_p_value"] <= all_models["two_sided_p_value"] else 1
                    ]
                ),
            },
            "neuroblastoma_subset_exploratory": {
                "n_models": int(len(nb)),
                "ln_ic50_spearman_rho": float(nb_spearman_ln.statistic),
                "ln_ic50_two_sided_p_value": float(nb_spearman_ln.pvalue),
                "auc_spearman_rho": float(nb_spearman_auc.statistic),
                "auc_two_sided_p_value": float(nb_spearman_auc.pvalue),
                "inferential_status": "underpowered exploratory subset; not confirmatory",
            },
        },
        "interpretation": (
            "Association is observational and does not establish a predictive biomarker, "
            "causal mechanism, clinical benefit, or novelty. The neuroblastoma subset "
            "is too small for confirmatory inference."
        ),
        "model_level_output": str(MODEL_OUTPUT.relative_to(ROOT)),
    }
    _write_json(SUMMARY_OUTPUT, summary)
    print(json.dumps(summary, indent=2))
    print(f"\nModel-level results: {MODEL_OUTPUT}")
    print(f"Summary: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
