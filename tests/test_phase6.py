import pytest

from src.ingestion.pdb_structural import build_phase6_dataset


def test_phase6_refuses_to_generate_unverified_examples(tmp_path):
    with pytest.raises(RuntimeError, match="hard-coded examples|generated TARGET sample IDs"):
        build_phase6_dataset(output_dir=tmp_path)

    assert not (tmp_path / "pdb_structural_phase6.csv").exists()
