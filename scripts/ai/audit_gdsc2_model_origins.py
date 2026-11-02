#!/usr/bin/env python3
"""Audit GDSC2 cell-line origin and donor age using Cellosaurus stable IDs."""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[2]
MODEL_PREDICTIONS = (
    ROOT
    / "data/ai/molecular_models/"
    "gdsc2_pediatric_associated_expression_ridge_ln_ic50_oof_predictions.csv"
)
OUTPUT_PATH = ROOT / "reports/gdsc2_pediatric_model_origin_audit.csv"
CELL0SAURUS_SEARCH_URL = "https://api.cellosaurus.org/search/cell-line"
CELL0SAURUS_RECORD_URL = "https://www.cellosaurus.org/"
CELL_MODEL_PASSPORT_DATABASE = "Cell_Model_Passport"


def parse_age_years(age: str) -> float | None:
    """Convert Cellosaurus age strings such as 1Y2M to years."""
    normalized = age.strip().upper()
    if not normalized:
        return None
    parts = re.findall(r"(\d+(?:\.\d+)?)([YMWD])", normalized)
    if not parts or "".join(value + unit for value, unit in parts) != normalized:
        return None
    conversions = {"Y": 1.0, "M": 1 / 12, "W": 7 / 365.25, "D": 1 / 365.25}
    return sum(float(value) * conversions[unit] for value, unit in parts)


def _records_for_model(
    session: requests.Session,
    cell_line: str,
    sanger_model_id: str,
) -> list[dict[str, Any]]:
    response = session.get(
        CELL0SAURUS_SEARCH_URL,
        params={
            "q": f'id:"{cell_line}"',
            "format": "json",
            "rows": 100,
            "fields": "id,ac,ag,di,dr",
        },
        timeout=(15, 60),
    )
    response.raise_for_status()
    payload = response.json()
    records = payload.get("Cellosaurus", {}).get("cell-line-list", [])
    matches = []
    for record in records:
        passport_ids = {
            xref.get("accession")
            for xref in record.get("xref-list", [])
            if xref.get("database") == CELL_MODEL_PASSPORT_DATABASE
        }
        if sanger_model_id in passport_ids:
            matches.append(record)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one Cellosaurus record for {cell_line} "
            f"({sanger_model_id}); found {len(matches)}."
        )
    return matches


def _audit_record(
    reported_cancer_type: str,
    cell_line: str,
    sanger_model_id: str,
    record: dict[str, Any],
    audited_at: str,
) -> dict[str, Any]:
    accession_list = record.get("accession-list", [])
    primary_accessions = [
        item.get("value")
        for item in accession_list
        if item.get("type") == "primary" and item.get("value")
    ]
    if len(primary_accessions) != 1:
        raise ValueError(
            f"Expected one primary Cellosaurus accession for {cell_line} "
            f"({sanger_model_id}); found {primary_accessions!r}."
        )

    age = str(record.get("age", "")).strip()
    age_years = parse_age_years(age)
    if age_years is None:
        classification = "age_unresolved"
    elif age_years < 18:
        classification = "confirmed_pediatric"
    else:
        classification = "confirmed_adult"

    diseases = sorted(
        {
            str(disease.get("label", "")).strip()
            for disease in record.get("disease-list", [])
            if disease.get("label")
        }
    )
    cellosaurus_id = primary_accessions[0]
    return {
        "reported_cancer_type": reported_cancer_type,
        "cell_line": cell_line,
        "sanger_model_id": sanger_model_id,
        "cellosaurus_id": cellosaurus_id,
        "cellosaurus_age": age,
        "age_years": "" if age_years is None else f"{age_years:.6f}",
        "origin_classification": classification,
        "cellosaurus_diseases": " | ".join(diseases),
        "cellosaurus_url": f"{CELL0SAURUS_RECORD_URL}{cellosaurus_id}",
        "cell_model_passport_url": (
            f"https://cellmodelpassports.sanger.ac.uk/passports/{sanger_model_id}"
        ),
        "audited_at_utc": audited_at,
    }


def audit_origins(
    model_predictions: Path = MODEL_PREDICTIONS,
    output_path: Path = OUTPUT_PATH,
) -> list[dict[str, Any]]:
    with model_predictions.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"reported_cancer_type", "cell_line", "sanger_model_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Prediction file is missing required columns: {sorted(required)}."
            )
        predictions = {
            (
                row["reported_cancer_type"].strip(),
                row["cell_line"].strip(),
                row["sanger_model_id"].strip(),
            )
            for row in reader
        }
    if not predictions:
        raise ValueError(f"No model identities found in {model_predictions}.")
    if any(not cancer or not line or not model_id for cancer, line, model_id in predictions):
        raise ValueError("Model predictions contain missing cell-line identities.")
    if len({model_id for _, _, model_id in predictions}) != len(predictions):
        raise ValueError("A Sanger model ID maps to multiple cell-line records.")

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "pediatric-drug-repurposing-research/0.1"}
    )
    audited_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for cancer_type, cell_line, sanger_model_id in sorted(predictions):
        records = _records_for_model(
            session,
            cell_line,
            sanger_model_id,
        )
        rows.append(
            _audit_record(
                cancer_type,
                cell_line,
                sanger_model_id,
                records[0],
                audited_at,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row["origin_classification"])
        counts[label] = counts.get(label, 0) + 1
    print(
        f"Audited {len(rows)} models; classifications={counts}; "
        f"saved {output_path.relative_to(ROOT)}"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=MODEL_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    audit_origins(args.predictions, args.output)


if __name__ == "__main__":
    main()
