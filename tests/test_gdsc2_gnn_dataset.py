import numpy as np
import pandas as pd
import pytest

from scripts.ai.prepare_gdsc2_gnn_dataset import assemble_response_dataset


def _fixture():
    responses = pd.DataFrame(
        [
            ("C1", "1", "Drug A", "Cell A", "SIDM001", "NB", 1, 1.2, 0.3),
            ("C2", "2", "Drug B", "Cell A", "SIDM001", "NB", 1, 2.2, 0.4),
            ("C3", "1", "Drug A", "Cell B", "SIDM002", "ALL", 2, 3.2, 0.5),
            ("C4", "2", "Drug B", "Cell B", "SIDM002", "ALL", 2, 4.2, 0.6),
        ],
        columns=[
            "curve_id", "drug_id", "drug_name", "cell_line", "sanger_model_id",
            "reported_cancer_type", "fold", "ln_ic50", "auc",
        ],
    )
    crosswalk = pd.DataFrame(
        [
            ("SIDM001", "Cell A", "GDSC_MODEL:SIDM001", True, 0,
             "Cell A, neuroblastoma", "matched_exact_casefold"),
            ("SIDM002", "Cell B", "GDSC_MODEL:SIDM002", True, 1,
             "Cell B, leukaemia", "matched_exact_casefold"),
            ("SIDM003", "Cell C", "GDSC_MODEL:SIDM003", True, np.nan,
             "", "unmatched"),
        ],
        columns=[
            "sanger_model_id", "gdsc_cell_line", "graph_node_id",
            "graph_node_present", "expression_sample_column_index_0based",
            "expression_sample_label", "match_status",
        ],
    )
    labels = ["Cell A, neuroblastoma", "Cell B, leukaemia"]
    genes = ["ENSG000001", "ENSG000002"]
    expression = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    return responses, crosswalk, labels, genes, expression


def test_assembles_only_exactly_matched_response_pairs_and_expression_rows():
    args = _fixture()
    pairs, features, model_ids, columns = assemble_response_dataset(*args)

    assert len(pairs) == 4
    assert pairs["sanger_model_id"].nunique() == 2
    assert set(pairs["curve_id"]) == {"C1", "C2", "C3", "C4"}
    assert model_ids == ["SIDM001", "SIDM002"]
    assert columns == [0, 1]
    assert features.shape == (2, 2)
    assert pairs["expression_row_index"].map({0: "SIDM001", 1: "SIDM002"}).notna().all()
    assert {"ln_ic50", "auc"}.issubset(pairs.columns)


def test_rejects_crosswalk_expression_label_mismatch():
    responses, crosswalk, labels, genes, expression = _fixture()
    crosswalk.loc[0, "expression_sample_label"] = "Wrong Cell, tumour"
    with pytest.raises(ValueError, match="sample label changed"):
        assemble_response_dataset(responses, crosswalk, labels, genes, expression)


def test_rejects_response_cell_name_mismatch():
    responses, crosswalk, labels, genes, expression = _fixture()
    responses.loc[0, "cell_line"] = "Different Cell"
    with pytest.raises(ValueError, match="disagree with the audited crosswalk"):
        assemble_response_dataset(responses, crosswalk, labels, genes, expression)


def test_rejects_model_assigned_to_multiple_folds():
    responses, crosswalk, labels, genes, expression = _fixture()
    responses.loc[1, "fold"] = 2
    with pytest.raises(ValueError, match="more than one held-out fold"):
        assemble_response_dataset(responses, crosswalk, labels, genes, expression)


def test_rejects_duplicate_expression_column_assignment():
    responses, crosswalk, labels, genes, expression = _fixture()
    crosswalk.loc[1, "expression_sample_column_index_0based"] = 0
    with pytest.raises(ValueError, match="one expression column to multiple models"):
        assemble_response_dataset(responses, crosswalk, labels, genes, expression)


def test_rejects_fold_id_outside_frozen_five_fold_scheme():
    responses, crosswalk, labels, genes, expression = _fixture()
    responses.loc[2, "fold"] = 6
    with pytest.raises(ValueError, match="Unexpected frozen fold IDs outside 1-5"):
        assemble_response_dataset(responses, crosswalk, labels, genes, expression)
