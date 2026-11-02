from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from src.utils.config import CONFIG

MAF_FILENAME = "snv-consensus-plus-hotspots.maf.tsv.gz"
MAF_COLUMNS = {
    "sample_id": "Tumor_Sample_Barcode",
    "gene_symbol": "Hugo_Symbol",
    "variant_classification": "Variant_Classification",
    "variant_type": "Variant_Type",
    "chromosome": "Chromosome",
    "start_position": "Start_Position",
    "reference_allele": "Reference_Allele",
    "tumor_allele": "Tumor_Seq_Allele2",
    "protein_change": "HGVSp_Short",
    "impact": "IMPACT",
}


def _resolve_maf_path(raw_dir: str | Path | None) -> Path:
    configured_path = os.environ.get("OPENPEDCAN_MAF_PATH")
    if configured_path:
        path = Path(configured_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"OPENPEDCAN_MAF_PATH does not point to a file: {path}")
        return path

    base_dir = Path(raw_dir) if raw_dir is not None else Path(CONFIG["raw_data_path"]) / "openpedcan"
    candidates = list(base_dir.rglob(MAF_FILENAME)) if base_dir.exists() else []
    candidates = [path for path in candidates if path.is_file()]
    candidates.sort(key=lambda path: ("testing" in {part.lower() for part in path.parts}, str(path)))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"{MAF_FILENAME} was not found under {base_dir}. Set OPENPEDCAN_MAF_PATH to a real OpenPedCan release file."
    )


def build_openpedcan_mutation_dataset(
    output_dir: str | Path | None = None,
    raw_dir: str | Path | None = None,
) -> Path:
    dataset_dir = Path(output_dir) if output_dir is not None else Path(CONFIG["raw_data_path"]) / "openpedcan"
    clinical_path = dataset_dir / "openpedcan_phase1.csv"
    clinical_manifest_path = dataset_dir / "openpedcan_manifest.json"
    if not clinical_path.is_file() or not clinical_manifest_path.is_file():
        raise FileNotFoundError("Build the real OpenPedCan Phase 1 clinical dataset before ingesting mutations")

    clinical_manifest = json.loads(clinical_manifest_path.read_text())
    if clinical_manifest.get("data_origin") != "OpenPedCan release data":
        raise ValueError("Mutation ingestion requires real OpenPedCan clinical data; synthetic fallback is not accepted")

    maf_path = _resolve_maf_path(raw_dir)
    clinical_metadata = pd.read_csv(
        clinical_path,
        usecols=["sample_id", "cohort", "tumor_type", "diagnosis"],
        dtype=str,
        keep_default_na=False,
    )
    clinical_samples = set(clinical_metadata["sample_id"])
    metadata_by_sample = {
        column: clinical_metadata.set_index("sample_id")[column]
        for column in ("cohort", "tumor_type", "diagnosis")
    }
    output_path = dataset_dir / "openpedcan_mutations_phase2.csv"
    temporary_path = output_path.with_suffix(".csv.tmp")
    input_rows = 0
    output_rows = 0
    input_samples: set[str] = set()
    matched_samples: set[str] = set()
    first_chunk = True

    try:
        chunks = pd.read_csv(
            maf_path,
            sep="\t",
            compression="gzip" if maf_path.name.endswith(".gz") else None,
            usecols=list(MAF_COLUMNS.values()),
            dtype=str,
            keep_default_na=False,
            chunksize=100_000,
        )
        for chunk in chunks:
            input_rows += len(chunk)
            chunk = chunk.rename(columns={source: target for target, source in MAF_COLUMNS.items()})
            chunk["sample_id"] = chunk["sample_id"].str.strip()
            chunk["gene_symbol"] = chunk["gene_symbol"].str.strip()
            input_samples.update(chunk.loc[chunk["sample_id"].ne(""), "sample_id"].unique())
            matched = chunk["sample_id"].isin(clinical_samples) & chunk["gene_symbol"].ne("")
            selected = chunk.loc[matched, list(MAF_COLUMNS)].copy()
            if selected.empty:
                continue

            matched_samples.update(selected["sample_id"].unique())
            for column, metadata in metadata_by_sample.items():
                selected[column] = selected["sample_id"].map(metadata).fillna("")
            selected["source"] = "OpenPedCan"
            selected.to_csv(temporary_path, mode="w" if first_chunk else "a", header=first_chunk, index=False)
            output_rows += len(selected)
            first_chunk = False

        if output_rows == 0:
            raise ValueError("The MAF contained no gene variants matching the real OpenPedCan biospecimen IDs")
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    manifest = {
        "source": "OpenPedCan consensus somatic variants",
        "data_origin": "OpenPedCan release data",
        "release_scope": "testing_subset" if "testing" in maf_path.parts else "release",
        "source_location": str(maf_path),
        "rows_read": input_rows,
        "variant_rows_matched": output_rows,
        "unique_biospecimens_matched": len(matched_samples),
        "unmatched_biospecimen_ids": len(input_samples - clinical_samples),
        "columns": [*MAF_COLUMNS, "cohort", "tumor_type", "diagnosis", "source"],
        "output_path": str(output_path),
        "phase": "phase_2",
    }
    (dataset_dir / "openpedcan_mutations_manifest.json").write_text(json.dumps(manifest, indent=2))
    return output_path