import pytest

from src.ingestion.bindingdb_go_opentargets import build_phase5_dataset


def test_phase5_refuses_to_generate_unverified_examples(tmp_path):
    with pytest.raises(RuntimeError, match="hard-coded examples|generated TARGET sample IDs"):
        build_phase5_dataset(output_dir=tmp_path)

    assert not (tmp_path / "bindingdb_go_opentargets_phase5.csv").exists()
