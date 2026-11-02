#!/usr/bin/env python3
"""Explore the fixed proteasome score against independent CTRP Bortezomib response."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

from scripts.ai.analyze_bortezomib_proteasome_expression import (
    EXPRESSION_URL,
    _read_expression_rows,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "reports"
MODEL_OUTPUT = OUTPUT_DIR / "bortezomib_ctrp_proteasome_expression_models.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "bortezomib_ctrp_proteasome_expression_summary.json"
CTRP_POINTS_URL = "https://portals.broadinstitute.org/ctrp.v2/rest/aucCorrelationPoints/"
CTRP_COMPOUND_ID = 411737
CTRP_FEATURE_GENES = ("PSMA1", "ACTB")
BOOTSTRAP_ITERATIONS = 20_000
RANDOM_SEED = 20261102
CELL0SAURUS_RECORD_URL = "https://www.cellosaurus.org/"

CTRP_MODEL_CROSSWALK = {
    "CHP126": ("CVCL_1123", "SIDM00286"),
    "IMR32": ("CVCL_0346", "SIDM00226"),
    "KELLY": ("CVCL_2092", "SIDM01009"),
    "KPNSI9S": ("CVCL_1340", "SIDM00016"),
    "KPNYN": ("CVCL_1341", "SIDM00556"),
    "MHHNB11": ("CVCL_1412", "SIDM00387"),
    "NB1": ("CVCL_1440", "SIDM00578"),
    "NH6": ("CVCL_1606", "SIDM01391"),
    "SIMA": ("CVCL_1695", "SIDM00395"),
    "SKNBE2": ("CVCL_0528", "SIDM00894"),
    "SKNFI": ("CVCL_1702", "SIDM01099"),
    "SKNSH": ("CVCL_0531", "SIDM01098"),
}

EXPRESSION_MODEL_CROSSWALK = {
    "CHP-126": ("CVCL_1123", "SIDM00286"),
    "CHP-212": ("CVCL_1125", "SIDM00910"),
    "IMR-32": ("CVCL_0346", "SIDM00226"),
    "KELLY": ("CVCL_2092", "SIDM01009"),
    "KP-N-RT-BM-1": ("CVCL_1339", "SIDM00557"),
    "KP-N-SI9s": ("CVCL_1340", "SIDM00016"),
    "KP-N-YN": ("CVCL_1341", "SIDM00556"),
    "MHH-NB-11": ("CVCL_1412", "SIDM00387"),
    "NB-1": ("CVCL_1440", "SIDM00578"),
    "SIMA": ("CVCL_1695", "SIDM00395"),
    "SK-N-AS": ("CVCL_1700", "SIDM01101"),
    "SK-N-BE(2)": ("CVCL_0528", "SIDM00894"),
    "SK-N-DZ": ("CVCL_1701", "SIDM01100"),
    "SK-N-FI": ("CVCL_1702", "SIDM01099"),
    "SK-N-SH": ("CVCL_0531", "SIDM01098"),
}


def _fetch_ctrp_points(gene: str) -> dict[str, dict[str, Any]]:
    payload = {
        "cpd_id": {
            "compound_id": CTRP_COMPOUND_ID,
            "compound_name": "bortezomib",
        },
        "gene_primary_name": gene,
        "dataset": "GEX",
        "filter": {
            "site_primary": "",
            "growth_mode": "",
            "feature_dataset": "",
        },
    }
    response = requests.post(CTRP_POINTS_URL, json=payload, timeout=(15, 60))
    response.raise_for_status()
    points = response.json()
    if not isinstance(points, list):
        raise ValueError("CTRP response did not contain a list of correlation points.")

    autonomic_ganglia = [
        point
        for point in points
        if point.get("sitePrimary") == "autonomic_ganglia"
    ]
    if not autonomic_ganglia:
        raise ValueError("CTRP returned no Bortezomib response points for autonomic ganglia.")

    by_name: dict[str, dict[str, Any]] = {}
    for point in autonomic_ganglia:
        name = str(point.get("cellName", "")).strip()
        if not name or name in by_name:
            raise ValueError(f"CTRP contains an empty or duplicate model name: {name!r}.")
        if point.get("cpdAUC") is None:
            raise ValueError(f"CTRP response is missing AUC for model {name!r}.")
        by_name[name] = point
    if set(by_name) != set(CTRP_MODEL_CROSSWALK):
        raise ValueError(
            "The CTRP autonomic-ganglia model set changed: "
            f"{sorted(by_name)}."
        )
    return by_name


def _bootstrap_spearman_ci(
    scores: np.ndarray, auc_values: np.ndarray
) -> tuple[float, float, int]:
    rng = np.random.default_rng(RANDOM_SEED)
    sample_count = len(scores)
    correlations: list[float] = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        indices = rng.integers(0, sample_count, size=sample_count)
        result = spearmanr(scores[indices], auc_values[indices])
        if np.isfinite(result.statistic):
            correlations.append(float(result.statistic))
    if not correlations:
        raise ValueError("No valid bootstrap Spearman correlations were generated.")
    lower, upper = np.quantile(correlations, [0.025, 0.975])
    return float(lower), float(upper), len(correlations)


def main() -> None:
    sample_labels, tpm_by_gene, expression_rows_sha256 = _read_expression_rows()
    expression_names = [
        label.rpartition(", ")[0].strip()
        for label in sample_labels
        if label.rpartition(", ")[2].strip().casefold() == "neuroblastoma"
    ]
    if not expression_names:
        raise ValueError("Expression Atlas returned no Neuroblastoma-labeled samples.")

    expression_frame = pd.DataFrame(tpm_by_gene, index=sample_labels)
    log_expression = np.log2(expression_frame + 1.0)
    standard_deviation = log_expression.std(axis=0, ddof=0)
    if (standard_deviation == 0).any():
        raise ValueError("A proteasome gene has constant expression across the cohort.")
    scores = ((log_expression - log_expression.mean(axis=0)) / standard_deviation).mean(
        axis=1
    )

    ctrp_by_name = _fetch_ctrp_points(CTRP_FEATURE_GENES[0])
    control_points = _fetch_ctrp_points(CTRP_FEATURE_GENES[1])
    ctrp_auc = {
        name: float(point["cpdAUC"]) for name, point in ctrp_by_name.items()
    }
    control_auc = {
        name: float(point["cpdAUC"]) for name, point in control_points.items()
    }
    if ctrp_auc != control_auc:
        raise ValueError("CTRP response AUCs changed with the queried expression gene.")

    ctrp_models = []
    for ctrp_name, point in sorted(ctrp_by_name.items()):
        cellosaurus_id, sanger_model_id = CTRP_MODEL_CROSSWALK[ctrp_name]
        ctrp_models.append(
            {
                "ctrp_cell_name": ctrp_name,
                "ctrp_auc": float(point["cpdAUC"]),
                "cellosaurus_id": cellosaurus_id,
                "sanger_model_id": sanger_model_id,
                "cellosaurus_record_url": (
                    f"{CELL0SAURUS_RECORD_URL}{cellosaurus_id}"
                ),
            }
        )

    expression_crosswalks = []
    for label in sample_labels:
        if label.rpartition(", ")[2].strip().casefold() != "neuroblastoma":
            continue
        name = label.rpartition(", ")[0].strip()
        if name not in EXPRESSION_MODEL_CROSSWALK:
            raise ValueError(f"No audited Cellosaurus crosswalk for expression sample {name!r}.")
        cellosaurus_id, sanger_model_id = EXPRESSION_MODEL_CROSSWALK[name]
        expression_crosswalks.append(
            {
                "expression_label": label,
                "expression_cell_name": name,
                "cellosaurus_id": cellosaurus_id,
                "sanger_model_id": sanger_model_id,
                "cellosaurus_record_url": (
                    f"{CELL0SAURUS_RECORD_URL}{cellosaurus_id}"
                ),
            }
        )

    expression_by_cellosaurus: dict[str, dict[str, Any]] = {}
    for item in expression_crosswalks:
        cellosaurus_id = item["cellosaurus_id"]
        if cellosaurus_id in expression_by_cellosaurus:
            raise ValueError(
                f"Multiple expression samples resolve to {cellosaurus_id}."
            )
        expression_by_cellosaurus[cellosaurus_id] = item

    matched_rows = []
    for model in ctrp_models:
        expression_model = expression_by_cellosaurus.get(model["cellosaurus_id"])
        if expression_model is None:
            continue
        matched_rows.append(
            {
                **model,
                "expression_label": expression_model["expression_label"],
                "proteasome_expression_score": float(
                    scores[expression_model["expression_label"]]
                ),
                "match_method": "shared_Cellosaurus_accession",
            }
        )

    if len(matched_rows) != 11:
        raise ValueError(
            f"Expected 11 stable-ID CTRP/expression matches; found {len(matched_rows)}."
        )
    matched = pd.DataFrame(matched_rows).sort_values("ctrp_cell_name")
    score_values = matched["proteasome_expression_score"].to_numpy(dtype=float)
    auc_values = matched["ctrp_auc"].to_numpy(dtype=float)
    association = spearmanr(score_values, auc_values)
    ci_low, ci_high, bootstrap_valid = _bootstrap_spearman_ci(score_values, auc_values)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matched.to_csv(MODEL_OUTPUT, index=False)
    summary = {
        "analysis": "Exploratory external-response test of the prespecified 38-gene score",
        "response_source": {
            "dataset": "Broad Cancer Therapeutics Response Portal (CTRP)",
            "portal_url": "https://portals.broadinstitute.org/ctrp.v2/",
            "points_endpoint": CTRP_POINTS_URL,
            "compound_id": CTRP_COMPOUND_ID,
            "compound_name": "Bortezomib",
            "response_metric": "CTRP cpdAUC; not directly comparable in scale to GDSC2 AUC",
            "response_gene_independence_check": list(CTRP_FEATURE_GENES),
            "auc_maps_identical_for_features": True,
            "site_filter": "autonomic_ganglia",
        },
        "expression_source": {
            "study": "E-MTAB-2770",
            "download_url": EXPRESSION_URL,
            "transform": "log2(TPM + 1), gene-wise z-score across all 1,019 samples",
            "score": "unweighted mean of the 38 gene-wise z-scores",
            "selected_gene_rows_sha256": expression_rows_sha256,
        },
        "identity_crosswalk": {
            "source": "Audited Cellosaurus records",
            "record_base_url": CELL0SAURUS_RECORD_URL,
            "join_key": "shared Cellosaurus accession",
            "stable_model_id": "Sanger Cell Model Passport ID recorded by Cellosaurus",
            "CTRP_neuroblastoma_response_models": len(ctrp_models),
            "expression_neuroblastoma_samples": len(expression_crosswalks),
            "stable_id_matched_models": len(matched_rows),
            "unmatched_ctrp_models": sorted(
                set(ctrp_model["ctrp_cell_name"] for ctrp_model in ctrp_models)
                - set(matched["ctrp_cell_name"])
            ),
            "model_level_output": str(MODEL_OUTPUT.relative_to(ROOT)),
        },
        "primary_test": {
            "test": "two-sided Spearman rank correlation",
            "n_models": len(matched),
            "rho": float(association.statistic),
            "two_sided_p_value": float(association.pvalue),
            "bootstrap_95_percentile_ci": [ci_low, ci_high],
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_valid_iterations": bootstrap_valid,
            "bootstrap_seed": RANDOM_SEED,
            "inferential_status": (
                "exploratory and underpowered; not confirmatory or clinically predictive"
            ),
        },
        "interpretation": (
            "This is an exploratory association using an independent drug-response screen "
            "but the same CCLE expression cohort. The small Neuroblastoma sample cannot "
            "confirm a biomarker or support clinical inference."
        ),
    }
    SUMMARY_OUTPUT.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"\nModel-level results: {MODEL_OUTPUT}")
    print(f"Summary: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
