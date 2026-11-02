import pytest

from src.ingestion.target import build_phase2_dataset


def test_target_refuses_to_generate_unverified_examples(tmp_path):
    with pytest.raises(RuntimeError, match="hard-coded examples|generated TARGET sample IDs"):
        build_phase2_dataset(output_dir=tmp_path)

    assert not (tmp_path / "target_phase2.csv").exists()
