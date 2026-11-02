import pytest

from src.ingestion.experimental_validation import build_phase7_dataset


def test_phase7_refuses_to_generate_unverified_examples(tmp_path):
    with pytest.raises(RuntimeError, match="hard-coded examples|generated TARGET sample IDs"):
        build_phase7_dataset(output_dir=tmp_path)

    assert not (tmp_path / "experimental_validation_phase7.csv").exists()
