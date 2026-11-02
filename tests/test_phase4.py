import pytest

from src.ingestion.chembl_reactome_string import build_phase4_dataset


def test_phase4_refuses_to_generate_unverified_examples(tmp_path):
    with pytest.raises(RuntimeError, match="hard-coded examples"):
        build_phase4_dataset(output_dir=tmp_path)

    assert not (tmp_path / "chembl_reactome_string_phase4.csv").exists()
