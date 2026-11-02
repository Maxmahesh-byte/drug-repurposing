from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.openpedcan import build_phase1_dataset


def test_build_phase1_dataset_creates_standardized_output(tmp_path):
    output_dir = tmp_path / "openpedcan"
    release_dir = tmp_path / "release" / "v15"
    release_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "Kids_First_Biospecimen_ID": ["BS-001", "BS-002"],
            "sample_id": ["S-001", "S-002"],
            "cohort": ["TARGET", "OpenPBTA"],
            "cancer_group": ["leukemia", "brain tumor"],
            "integrated_diagnosis": ["B-ALL", "medulloblastoma"],
            "molecular_subtype": ["B-cell", "SHH"],
        }
    ).to_csv(release_dir / "histologies.tsv", sep="\t", index=False)

    df = build_phase1_dataset(output_dir=output_dir, raw_dir=release_dir)

    expected_columns = {
        "sample_id",
        "cohort",
        "tumor_type",
        "diagnosis",
        "molecular_subtype",
        "source",
    }

    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == expected_columns
    assert df["sample_id"].tolist() == ["BS-001", "BS-002"]
    assert df["diagnosis"].tolist() == ["B-ALL", "medulloblastoma"]
    assert df["molecular_subtype"].tolist() == ["B-cell", "SHH"]
    assert (output_dir / "openpedcan_phase1.csv").exists()
    manifest = (output_dir / "openpedcan_manifest.json").read_text()
    assert '"data_origin": "OpenPedCan release data"' in manifest
    assert '"source_location":' in manifest


def test_build_phase1_dataset_requires_real_data_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENPEDCAN_DATA_PATH", raising=False)
    monkeypatch.delenv("OPENPEDCAN_URL", raising=False)
    monkeypatch.delenv("OPENPEDCAN_ALLOW_SYNTHETIC_FALLBACK", raising=False)

    with pytest.raises(FileNotFoundError, match="histologies.tsv was not found"):
        build_phase1_dataset(output_dir=tmp_path / "output", raw_dir=tmp_path / "empty")


def test_synthetic_fallback_requires_explicit_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENPEDCAN_ALLOW_SYNTHETIC_FALLBACK", "1")

    df = build_phase1_dataset(output_dir=tmp_path / "output", raw_dir=tmp_path / "empty")

    assert set(df["cohort"]) == {"synthetic"}
    manifest = (tmp_path / "output" / "openpedcan_manifest.json").read_text()
    assert '"data_origin": "synthetic_fallback"' in manifest
