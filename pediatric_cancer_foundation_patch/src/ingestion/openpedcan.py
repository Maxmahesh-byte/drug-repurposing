from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DEFAULT_RELEASE = os.getenv("OPENPEDCAN_RELEASE", "v15")
DEFAULT_BASE_URL = os.getenv(
    "OPENPEDCAN_BASE_URL",
    "https://s3.amazonaws.com/d3b-openaccess-us-east-1-prd-pbta/open-targets",
)

CORE_FILES = (
    "histologies.tsv",
    "efo-mondo-map.tsv",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "openpedcan"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(destination)


def download_openpedcan(
    release: str = DEFAULT_RELEASE,
    base_url: str = DEFAULT_BASE_URL,
    files: tuple[str, ...] = CORE_FILES,
    force: bool = False,
) -> dict:
    release_dir = RAW_DIR / release
    manifest = {
        "source": "OpenPedCan",
        "release": release,
        "base_url": base_url.rstrip("/"),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [],
    }

    for filename in files:
        url = f"{base_url.rstrip('/')}/{release}/{filename}"
        destination = release_dir / filename

        if force or not destination.exists():
            print(f"Downloading {filename}")
            download_file(url, destination)
        else:
            print(f"Keeping existing {filename}")

        manifest["files"].append(
            {
                "filename": filename,
                "url": url,
                "path": str(destination.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(destination),
                "bytes": destination.stat().st_size,
            }
        )

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = METADATA_DIR / "openpedcan_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}")
    return manifest


def _snake_case(name: str) -> str:
    value = re.sub(r"[^0-9A-Za-z]+", "_", str(name).strip())
    value = re.sub(r"_+", "_", value).strip("_").lower()
    return value


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_snake_case(c) for c in out.columns]
    return out


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def standardize_histologies(
    input_path: Path | None = None,
    output_path: Path | None = None,
    exclude_cohorts: tuple[str, ...] = ("tcga", "gtex"),
) -> pd.DataFrame:
    release = DEFAULT_RELEASE
    if input_path is None:
        input_path = RAW_DIR / release / "histologies.tsv"
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "processed" / "pediatric_samples.tsv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"OpenPedCan histologies file not found: {input_path}. "
            "Run src/ingestion/data_ingest.py first."
        )

    df = pd.read_csv(input_path, sep="\t", dtype=str, low_memory=False)
    df = normalize_columns(df)

    sample_col = first_existing(
        df,
        [
            "kids_first_biospecimen_id",
            "sample_id",
            "sample",
            "biospecimen_id",
        ],
    )
    patient_col = first_existing(
        df,
        [
            "kids_first_participant_id",
            "participant_id",
            "patient_id",
            "case_id",
        ],
    )
    diagnosis_col = first_existing(
        df,
        [
            "harmonized_diagnosis",
            "integrated_diagnosis",
            "cancer_group",
            "diagnosis",
            "tumor_descriptor",
        ],
    )
    cohort_col = first_existing(df, ["cohort", "dataset", "source"])
    descriptor_col = first_existing(
        df,
        ["tumor_descriptor", "specimen_type", "sample_type"],
    )

    if sample_col is None:
        raise ValueError(
            "Could not identify a sample identifier column in histologies.tsv."
        )
    if diagnosis_col is None:
        raise ValueError(
            "Could not identify a diagnosis/cancer-group column in histologies.tsv."
        )

    out = pd.DataFrame(
        {
            "sample_id": df[sample_col].astype("string").str.strip(),
            "patient_id": (
                df[patient_col].astype("string").str.strip()
                if patient_col
                else pd.Series(pd.NA, index=df.index, dtype="string")
            ),
            "disease": df[diagnosis_col].astype("string").str.strip(),
            "cohort": (
                df[cohort_col].astype("string").str.strip()
                if cohort_col
                else pd.Series(pd.NA, index=df.index, dtype="string")
            ),
            "specimen_type": (
                df[descriptor_col].astype("string").str.strip()
                if descriptor_col
                else pd.Series(pd.NA, index=df.index, dtype="string")
            ),
        }
    )

    for col in ["sample_id", "patient_id", "disease", "cohort", "specimen_type"]:
        out[col] = out[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    if cohort_col:
        normalized_exclusions = {x.lower() for x in exclude_cohorts}
        out = out[
            ~out["cohort"].fillna("").str.lower().isin(normalized_exclusions)
        ].copy()

    out = out.dropna(subset=["sample_id", "disease"])
    out = out.drop_duplicates(subset=["sample_id"]).reset_index(drop=True)

    # Keep a few common molecular-data availability flags when present.
    for output_name, candidates in {
        "has_rnaseq": ["experimental_strategy", "rna_library"],
        "has_dna": ["experimental_strategy"],
    }.items():
        source_col = first_existing(df, candidates)
        if source_col is not None:
            # These are intentionally coarse availability flags.
            values = df.loc[out.index, source_col].astype("string")
            if output_name == "has_rnaseq":
                out[output_name] = values.str.contains(
                    "rna", case=False, na=False
                ).astype("int8")
            else:
                out[output_name] = values.str.contains(
                    "wgs|wxs|sequencing|target", case=False, na=False, regex=True
                ).astype("int8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, sep="\t", index=False)
    print(f"Wrote {len(out):,} standardized samples to {output_path}")
    return out


if __name__ == "__main__":
    download_openpedcan()
    standardize_histologies()
