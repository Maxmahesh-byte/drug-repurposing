import pandas as pd
import pytest

from scripts.ai.build_gdsc2_expression_crosswalk import build_crosswalk


def _responses(rows):
    return pd.DataFrame(
        rows,
        columns=["sanger_model_id", "cell_line", "reported_cancer_type"],
    )


def test_crosswalk_matches_case_insensitive_exact_name_and_graph_node():
    responses = _responses(
        [
            ("SIDM001", "Cell A", "NB"),
            ("SIDM001", "Cell A", "NB"),
            ("SIDM002", "Cell B", "ALL"),
        ]
    )
    labels = ["cell a, neuroblastoma", "Cell B, leukaemia"]
    graph_ids = {"GDSC_MODEL:SIDM001", "GDSC_MODEL:SIDM002"}

    result, summary = build_crosswalk(responses, labels, graph_ids)

    first = result.set_index("sanger_model_id").loc["SIDM001"]
    assert first["match_status"] == "matched_exact_casefold"
    assert first["expression_sample_column_index_0based"] == 0
    assert bool(first["graph_node_present"]) is True
    assert first["gdsc_response_rows"] == 2
    assert summary["matched_models"] == 2
    assert summary["unique_expression_columns_used"] == 2
    assert summary["status"] == "passed"


def test_crosswalk_marks_duplicate_expression_name_as_ambiguous():
    responses = _responses([("SIDM001", "Cell A", "NB")])
    labels = ["Cell A, neuroblastoma", "cell a, other diagnosis"]

    result, summary = build_crosswalk(
        responses, labels, {"GDSC_MODEL:SIDM001"}
    )

    assert result.iloc[0]["match_status"] == "ambiguous"
    assert pd.isna(result.iloc[0]["expression_sample_column_index_0based"])
    assert summary["ambiguous_models"] == 1
    assert summary["matched_models"] == 0


def test_crosswalk_marks_same_name_for_multiple_sanger_ids_ambiguous():
    responses = _responses(
        [
            ("SIDM001", "Shared Name", "NB"),
            ("SIDM002", "shared name", "MB"),
        ]
    )

    result, summary = build_crosswalk(
        responses, ["Shared Name, tumour"], 
        {"GDSC_MODEL:SIDM001", "GDSC_MODEL:SIDM002"}
    )

    assert set(result["match_status"]) == {"ambiguous"}
    assert summary["ambiguous_models"] == 2


def test_crosswalk_keeps_unmatched_name_and_reports_missing_graph_node():
    responses = _responses([("SIDM001", "Not In Atlas", "NB")])

    result, summary = build_crosswalk(responses, ["Other Cell, tumour"], set())

    assert result.iloc[0]["match_status"] == "unmatched"
    assert bool(result.iloc[0]["graph_node_present"]) is False
    assert summary["unmatched_models"] == 1
    assert summary["graph_cell_line_nodes_missing"] == 1
    assert summary["status"] == "warning"


def test_crosswalk_rejects_one_sanger_id_with_multiple_cell_line_names():
    responses = _responses(
        [
            ("SIDM001", "Cell A", "NB"),
            ("SIDM001", "Cell A alias", "NB"),
        ]
    )

    with pytest.raises(ValueError, match="maps to multiple GDSC2 cell-line names"):
        build_crosswalk(
            responses,
            ["Cell A, tumour"],
            {"GDSC_MODEL:SIDM001"},
        )


def test_crosswalk_rejects_two_models_assigned_same_expression_column():
    # This would require two different source names to casefold to the same
    # crosswalk key; the identity ambiguity must be surfaced, not silently mapped.
    responses = _responses(
        [
            ("SIDM001", "Cell A", "NB"),
            ("SIDM002", "cell a", "MB"),
        ]
    )

    result, summary = build_crosswalk(
        responses,
        ["Cell A, tumour"],
        {"GDSC_MODEL:SIDM001", "GDSC_MODEL:SIDM002"},
    )

    assert summary["ambiguous_models"] == 2
    assert result["expression_sample_column_index_0based"].isna().all()
