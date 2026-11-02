from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

import pandas as pd

from src.utils.config import CONFIG

CANONICAL_COLUMNS = [
    "sample_id",
    "cohort",
    "tumor_type",
    "diagnosis",
    "molecular_subtype",
    "source",
]

FALLBACK_ROWS = [
    {
        "sample_id": "PED-001",
        "cohort": "synthetic",
        "tumor_type": "medulloblastoma",
        "diagnosis": "Medulloblastoma",
        "molecular_subtype": "",
        "source": "OpenPedCan",
    },
    {
        "sample_id": "PED-002",
        "cohort": "synthetic",
        "tumor_type": "neuroblastoma",
        "diagnosis": "Neuroblastoma",
        "molecular_subtype": "",
        "source": "OpenPedCan",
    },
    {
        "sample_id": "PED-003",
        "cohort": "synthetic",
        "tumor_type": "osteosarcoma",
        "diagnosis": "Osteosarcoma",
        "molecular_subtype": "",
        "source": "OpenPedCan",
    },
    {
        "sample_id": "PED-004",
        "cohort": "synthetic",
        "tumor_type": "acute lymphoblastic leukemia",
        "diagnosis": "ALL",
        "molecular_subtype": "",
        "source": "OpenPedCan",
    },
]


def _safe_lower(value: object) -> str:
    return str(value).strip().lower() if value is not None else ""


def _read_table(path: Path) -> pd.DataFrame:
    separator = "," if ".csv" in path.name.lower() else "\t"
    return pd.read_csv(path, sep=separator, dtype=str, low_memory=False)


def _candidate_openpedcan_files(base_dir: Path) -> Iterable[Path]:
    if not base_dir.exists():
        return []
    return sorted(
        path
        for path in base_dir.rglob("*")
        if path.is_file() and path.name.lower() in {"histologies.tsv", "histologies.tsv.gz"}
    )


def load_openpedcan_dataset(raw_dir: str | Path | None = None) -> pd.DataFrame:
    base_dir = Path(raw_dir) if raw_dir is not None else Path(CONFIG["raw_data_path"]) / "openpedcan"
    env_path_value = os.environ.get("OPENPEDCAN_DATA_PATH")
    if env_path_value:
        env_path = Path(env_path_value).expanduser()
        if not env_path.is_file():
            raise FileNotFoundError(f"OPENPEDCAN_DATA_PATH does not point to a file: {env_path}")
        data = _read_table(env_path)
        data.attrs["data_origin"] = "OpenPedCan release data"
        data.attrs["source_location"] = str(env_path)
        return data

    candidate_files = list(_candidate_openpedcan_files(base_dir))
    if candidate_files:
        data = _read_table(candidate_files[0])
        data.attrs["data_origin"] = "OpenPedCan release data"
        data.attrs["source_location"] = str(candidate_files[0])
        return data

    configured_url = os.environ.get("OPENPEDCAN_URL")
    if configured_url:
        request = Request(configured_url, headers={"User-Agent": "drug-repurposing/1.0"})
        with urlopen(request, timeout=60) as response:
            content = response.read()
        separator = "," if ".csv" in configured_url.lower() else "\t"
        compression = "gzip" if configured_url.lower().endswith(".gz") else None
        data = pd.read_csv(
            pd.io.common.BytesIO(content),
            sep=separator,
            compression=compression,
            dtype=str,
            low_memory=False,
        )
        data.attrs["data_origin"] = "OpenPedCan release data"
        data.attrs["source_location"] = configured_url
        return data

    if os.environ.get("OPENPEDCAN_ALLOW_SYNTHETIC_FALLBACK", "").lower() in {"1", "true", "yes"}:
        print("WARNING: using synthetic OpenPedCan fallback rows; this is not real patient data.")
        data = pd.DataFrame(FALLBACK_ROWS)
        data.attrs["data_origin"] = "synthetic_fallback"
        return data

    raise FileNotFoundError(
        "OpenPedCan histologies.tsv was not found. Download an OpenPedCan release and either "
        "place histologies.tsv under data/raw/openpedcan/ or set OPENPEDCAN_DATA_PATH to its path. "
        "Set OPENPEDCAN_ALLOW_SYNTHETIC_FALLBACK=1 only for development."
    )


def standardize_openpedcan_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        standardized = pd.DataFrame(columns=CANONICAL_COLUMNS)
    else:
        df = df.copy()
        normalized_columns = { _safe_lower(col): col for col in df.columns }
        aliases = {
            "sample_id": ["kids_first_biospecimen_id", "biospecimen_id", "specimen_id", "sample_id", "sample", "case_id", "case", "tumor_sample_id", "kids_first_participant_id", "patient_id"],
            "cohort": ["cohort", "project", "study"],
            "tumor_type": ["tumor_type", "cancer_group", "cancer_type", "disease_type", "primary_site", "tumor", "cancer"],
            "diagnosis": ["integrated_diagnosis", "harmonized_diagnosis", "diagnosis", "disease", "tumor_name", "primary_diagnosis", "diagnosis_name"],
            "molecular_subtype": ["molecular_subtype", "subtype", "subgroup"],
            "source": ["source", "data_source", "dataset", "platform"],
        }

        rename_map = {}
        for canonical, options in aliases.items():
            for option in options:
                if option in normalized_columns:
                    rename_map[normalized_columns[option]] = canonical
                    break
        if "sample_id" not in rename_map.values():
            raise ValueError("OpenPedCan histologies table has no recognized biospecimen/sample ID column")
        if not {"tumor_type", "diagnosis"}.intersection(rename_map.values()):
            raise ValueError("OpenPedCan histologies table has no recognized tumor type or diagnosis column")

        for canonical in CANONICAL_COLUMNS:
            selected_source = next(
                (source for source, target in rename_map.items() if target == canonical),
                None,
            )
            if selected_source and canonical in df.columns and selected_source != canonical:
                df = df.drop(columns=[canonical])

        df = df.rename(columns=rename_map)
        for canonical in CANONICAL_COLUMNS:
            if canonical not in df.columns:
                df[canonical] = None
        standardized = df[CANONICAL_COLUMNS].copy()

    standardized = standardized.fillna({
        "sample_id": "",
        "cohort": "",
        "tumor_type": "",
        "diagnosis": "",
        "molecular_subtype": "",
        "source": "OpenPedCan",
    })
    for column in CANONICAL_COLUMNS:
        standardized[column] = standardized[column].fillna("").astype(str)
    return standardized


def build_phase1_dataset(output_dir: str | Path | None = None, raw_dir: str | Path | None = None) -> pd.DataFrame:
    dataset_dir = Path(output_dir) if output_dir is not None else Path(CONFIG["raw_data_path"]) / "openpedcan"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_openpedcan_dataset(raw_dir=raw_dir)
    data_origin = raw_df.attrs.get("data_origin", "OpenPedCan release data")
    source_location = raw_df.attrs.get("source_location")
    standardized = standardize_openpedcan_columns(raw_df)

    output_path = dataset_dir / "openpedcan_phase1.csv"
    standardized.to_csv(output_path, index=False)

    manifest = {
        "source": "OpenPedCan",
        "data_origin": data_origin,
        "source_location": source_location,
        "rows": int(len(standardized)),
        "columns": list(standardized.columns),
        "output_path": str(output_path),
        "phase": "phase_1",
    }
    (dataset_dir / "openpedcan_manifest.json").write_text(json.dumps(manifest, indent=2))

    return standardized


if __name__ == "__main__":
    build_phase1_dataset()
