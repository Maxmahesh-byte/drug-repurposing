from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from src.utils.config import CONFIG

GDSC2_FILENAME = "GDSC2_fitted_dose_response_27Oct23.xlsx"
GDSC2_COLUMNS = {
    "dataset": "DATASET",
    "result_id": "NLME_RESULT_ID",
    "curve_id": "NLME_CURVE_ID",
    "cosmic_id": "COSMIC_ID",
    "cell_line": "CELL_LINE_NAME",
    "sanger_model_id": "SANGER_MODEL_ID",
    "reported_cancer_type": "TCGA_DESC",
    "drug_id": "DRUG_ID",
    "drug_name": "DRUG_NAME",
    "putative_target": "PUTATIVE_TARGET",
    "pathway": "PATHWAY_NAME",
    "min_concentration": "MIN_CONC",
    "max_concentration": "MAX_CONC",
    "ln_ic50": "LN_IC50",
    "auc": "AUC",
    "rmse": "RMSE",
    "z_score": "Z_SCORE",
}


def _resolve_gdsc2_path(raw_path: str | Path | None) -> Path:
    configured = os.environ.get("GDSC2_DATA_PATH")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"GDSC2_DATA_PATH does not point to a file: {path}")
        return path

    configured_path = Path(raw_path).expanduser() if raw_path is not None else None
    if configured_path is not None and configured_path.is_file():
        return configured_path
    base_dir = configured_path if configured_path is not None else Path(CONFIG["raw_data_path"]) / "gdsc"
    path = base_dir / GDSC2_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Official GDSC2 workbook not found: {path}; set GDSC2_DATA_PATH")
    return path


def build_gdsc2_response_dataset(
    output_dir: str | Path | None = None,
    raw_path: str | Path | None = None,
) -> Path:
    dataset_dir = Path(output_dir) if output_dir is not None else Path(CONFIG["raw_data_path"]) / "gdsc"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    source_path = _resolve_gdsc2_path(raw_path)
    raw = pd.read_excel(source_path, sheet_name=0, usecols=list(GDSC2_COLUMNS.values()))
    if not raw["DATASET"].astype(str).str.upper().eq("GDSC2").all():
        raw = raw[raw["DATASET"].astype(str).str.upper().eq("GDSC2")].copy()
    if raw.empty:
        raise ValueError("The official workbook contains no GDSC2 response rows")

    responses = raw.rename(columns={source: target for target, source in GDSC2_COLUMNS.items()})
    for column in ("ln_ic50", "auc", "rmse", "z_score", "min_concentration", "max_concentration"):
        responses[column] = pd.to_numeric(responses[column], errors="coerce")
    responses = responses[responses["ln_ic50"].notna() & responses["auc"].notna()].copy()
    if responses.empty:
        raise ValueError("GDSC2 workbook has no rows with measured LN_IC50 and AUC")

    output_path = dataset_dir / "gdsc2_responses.csv"
    temporary_path = output_path.with_suffix(".csv.tmp")
    try:
        responses.to_csv(temporary_path, index=False)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    manifest = {
        "source": "Wellcome Sanger GDSC2",
        "data_origin": "official_gdsc2_release",
        "release": "2023-10-27",
        "source_location": str(source_path),
        "rows_read": len(raw),
        "response_rows": len(responses),
        "unique_cell_lines": int(responses["sanger_model_id"].nunique()),
        "unique_drugs": int(responses["drug_id"].nunique()),
        "reported_cancer_types": sorted(responses["reported_cancer_type"].dropna().astype(str).unique().tolist()),
        "columns": list(responses.columns),
        "output_path": str(output_path),
    }
    (dataset_dir / "gdsc2_manifest.json").write_text(json.dumps(manifest, indent=2))
    return output_path