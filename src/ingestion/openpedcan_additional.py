from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from src.utils.config import CONFIG

CNV_FILENAME = "consensus_wgs_plus_cnvkit_wxs_plus_freec_tumor_only.tsv.gz"
FUSION_FILENAME = "fusion-putative-oncogenic.tsv"
CLINICAL_FIELDS = ("cohort", "tumor_type", "diagnosis")
CNV_FIELDS = {
    "sample_id": "biospecimen_id",
    "gene_symbol": "gene_symbol",
    "status": "status",
    "copy_number": "copy_number",
    "ploidy": "ploidy",
    "cytoband": "cytoband",
}
FUSION_FIELDS = {
    "sample_id": "Sample",
    "fusion_name": "FusionName",
    "gene_a": "Gene1A",
    "gene_b": "Gene1B",
    "ensembl_a": "Gene2A",
    "ensembl_b": "Gene2B",
    "fusion_annotation": "Fusion_anno",
    "caller_count": "caller.count",
    "fusion_type": "Fusion_Type",
    "breakpoint_location": "BreakpointLocation",
}


def _resolve_source(raw_dir: str | Path | None, filename: str, env_var: str) -> Path:
    env_path = os.environ.get(env_var)
    if env_path:
        path = Path(env_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{env_var} does not point to a file: {path}")
        return path

    base_dir = Path(raw_dir) if raw_dir is not None else Path(CONFIG["raw_data_path"]) / "openpedcan"
    candidates = [path for path in base_dir.rglob(filename) if path.is_file()] if base_dir.exists() else []
    candidates.sort(key=lambda path: ("testing" in {part.lower() for part in path.parts}, str(path)))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"{filename} was not found under {base_dir}; set {env_var} to a real release file")


def _clinical_lookup(dataset_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    clinical_path = dataset_dir / "openpedcan_phase1.csv"
    manifest_path = dataset_dir / "openpedcan_manifest.json"
    if not clinical_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Build the real OpenPedCan Phase 1 dataset before ingesting molecular evidence")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("data_origin") != "OpenPedCan release data":
        raise ValueError("Molecular ingestion requires real OpenPedCan clinical data")
    clinical = pd.read_csv(
        clinical_path,
        usecols=["sample_id", *CLINICAL_FIELDS],
        dtype=str,
        keep_default_na=False,
    ).set_index("sample_id")
    return clinical, {field: clinical[field] for field in CLINICAL_FIELDS}


def build_openpedcan_copy_number_dataset(
    output_dir: str | Path | None = None,
    raw_dir: str | Path | None = None,
) -> Path:
    dataset_dir = Path(output_dir) if output_dir is not None else Path(CONFIG["raw_data_path"]) / "openpedcan"
    clinical, metadata = _clinical_lookup(dataset_dir)
    clinical_samples = set(clinical.index)
    source_path = _resolve_source(raw_dir, CNV_FILENAME, "OPENPEDCAN_CNV_PATH")
    output_path = dataset_dir / "openpedcan_copy_number_phase3.csv"
    temporary_path = output_path.with_suffix(".csv.tmp")
    input_rows = output_rows = 0
    input_samples: set[str] = set()
    matched_samples: set[str] = set()
    first_chunk = True

    try:
        chunks = pd.read_csv(
            source_path,
            sep="\t",
            compression="gzip" if source_path.name.endswith(".gz") else None,
            usecols=list(CNV_FIELDS.values()),
            dtype=str,
            keep_default_na=False,
            chunksize=100_000,
        )
        for chunk in chunks:
            input_rows += len(chunk)
            chunk = chunk.rename(columns={source: target for target, source in CNV_FIELDS.items()})
            chunk["sample_id"] = chunk["sample_id"].str.strip()
            chunk["gene_symbol"] = chunk["gene_symbol"].str.strip()
            input_samples.update(chunk.loc[chunk["sample_id"].ne(""), "sample_id"].unique())
            non_neutral = chunk["status"].str.strip().str.lower().ne("neutral") & chunk["status"].str.strip().ne("")
            matched = chunk["sample_id"].isin(clinical_samples) & chunk["gene_symbol"].ne("") & non_neutral
            selected = chunk.loc[matched, list(CNV_FIELDS)].copy()
            if selected.empty:
                continue
            matched_samples.update(selected["sample_id"].unique())
            for field, values in metadata.items():
                selected[field] = selected["sample_id"].map(values).fillna("")
            selected["source"] = "OpenPedCan"
            selected.to_csv(temporary_path, mode="w" if first_chunk else "a", header=first_chunk, index=False)
            output_rows += len(selected)
            first_chunk = False

        if output_rows == 0:
            raise ValueError("No non-neutral copy-number rows matched real OpenPedCan biospecimens")
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    manifest = {
        "source": "OpenPedCan gene-level copy-number calls",
        "data_origin": "OpenPedCan release data",
        "release_scope": "testing_subset" if "testing" in source_path.parts else "release",
        "source_location": str(source_path),
        "rows_read": input_rows,
        "altered_gene_rows_matched": output_rows,
        "unique_biospecimens_matched": len(matched_samples),
        "unmatched_biospecimen_ids": len(input_samples - clinical_samples),
        "neutral_rows_excluded": True,
        "columns": [*CNV_FIELDS, *CLINICAL_FIELDS, "source"],
        "output_path": str(output_path),
        "phase": "phase_3",
    }
    (dataset_dir / "openpedcan_copy_number_manifest.json").write_text(json.dumps(manifest, indent=2))
    return output_path


def build_openpedcan_fusion_dataset(
    output_dir: str | Path | None = None,
    raw_dir: str | Path | None = None,
) -> Path:
    dataset_dir = Path(output_dir) if output_dir is not None else Path(CONFIG["raw_data_path"]) / "openpedcan"
    clinical, metadata = _clinical_lookup(dataset_dir)
    clinical_samples = set(clinical.index)
    source_path = _resolve_source(raw_dir, FUSION_FILENAME, "OPENPEDCAN_FUSION_PATH")
    fusions = pd.read_csv(
        source_path,
        sep="\t",
        usecols=list(FUSION_FIELDS.values()),
        dtype=str,
        keep_default_na=False,
    )
    rows_read = len(fusions)
    fusions = fusions.rename(columns={source: target for target, source in FUSION_FIELDS.items()})
    fusions["sample_id"] = fusions["sample_id"].str.strip()
    input_samples = set(fusions.loc[fusions["sample_id"].ne(""), "sample_id"])
    fusions = fusions[fusions["sample_id"].isin(clinical_samples) & fusions["fusion_name"].ne("")].copy()
    if fusions.empty:
        raise ValueError("No filtered fusion rows matched real OpenPedCan biospecimens")

    for field, values in metadata.items():
        fusions[field] = fusions["sample_id"].map(values).fillna("")
    fusions["source"] = "OpenPedCan"
    output_columns = [*FUSION_FIELDS, *CLINICAL_FIELDS, "source"]
    output_path = dataset_dir / "openpedcan_fusions_phase4.csv"
    temporary_path = output_path.with_suffix(".csv.tmp")
    try:
        fusions[output_columns].to_csv(temporary_path, index=False)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    manifest = {
        "source": "OpenPedCan putative oncogenic fusions",
        "data_origin": "OpenPedCan release data",
        "release_scope": "testing_subset" if "testing" in source_path.parts else "release",
        "source_location": str(source_path),
        "rows_read": rows_read,
        "fusion_rows_matched": len(fusions),
        "unique_biospecimens_matched": int(fusions["sample_id"].nunique()),
        "unmatched_biospecimen_ids": len(input_samples - clinical_samples),
        "columns": output_columns,
        "output_path": str(output_path),
        "phase": "phase_4",
    }
    (dataset_dir / "openpedcan_fusions_manifest.json").write_text(json.dumps(manifest, indent=2))
    return output_path