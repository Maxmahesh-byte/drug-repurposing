"""Prediction helpers for the trained GDSC2 expression-response model."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_expression_model(path: str | Path) -> dict[str, Any]:
    """Load a versioned JSON model bundle without deserializing executable objects."""
    bundle_path = Path(path)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("schema_version") != 1 or not isinstance(bundle.get("targets"), dict):
        raise ValueError(f"Unsupported expression model bundle: {bundle_path}")
    return bundle


def predict_expression_response(
    bundle: dict[str, Any],
    drug_id: str,
    tpm_by_gene: dict[str, float],
) -> dict[str, Any]:
    """Predict GDSC2 LN(IC50) and AUC for one known drug from gene-level TPM."""
    if not isinstance(tpm_by_gene, dict):
        raise TypeError("tpm_by_gene must map gene IDs to raw TPM values.")

    predictions: dict[str, float] = {}
    training_models: dict[str, int] = {}
    for endpoint, models in bundle["targets"].items():
        model = models.get(str(drug_id))
        if model is None:
            continue
        feature_ids = model["feature_ids"]
        missing = [gene_id for gene_id in feature_ids if gene_id not in tpm_by_gene]
        if missing:
            raise ValueError(
                f"Expression input is missing {len(missing)} required model features."
            )
        try:
            tpm = np.asarray([float(tpm_by_gene[gene_id]) for gene_id in feature_ids])
        except (TypeError, ValueError) as error:
            raise ValueError("Expression TPM values must be numeric.") from error
        if not np.isfinite(tpm).all() or np.any(tpm < 0):
            raise ValueError("Expression TPM values must be finite and non-negative.")
        values = np.log2(tpm + 1.0)
        standardized = (values - np.asarray(model["feature_means"])) / np.asarray(
            model["feature_scales"]
        )
        prediction = float(np.dot(standardized, model["coefficients"]) + model["intercept"])
        if not math.isfinite(prediction):
            raise ValueError(f"Model produced a non-finite {endpoint} prediction.")
        predictions[endpoint] = prediction
        training_models[endpoint] = int(model["training_models"])

    if not predictions:
        raise KeyError(f"No trained expression model exists for GDSC2 drug {drug_id!r}.")
    return {
        "drug_id": str(drug_id),
        "predictions": predictions,
        "training_models": training_models,
        "interpretation": (
            "GDSC2 cell-line response estimate for a known drug; not a patient or "
            "clinical-response prediction."
        ),
    }
