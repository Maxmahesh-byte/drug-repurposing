import json
import zipfile

import pandas as pd

from src.ingestion.nci_almanac import (
    _canonical_drug_pair,
    _gdsc_drug_crosswalk,
    _normalize_study_combo_score,
    build_nci_almanac_datasets,
)
from src.ingestion.validate_nci_almanac import (
    _ScoreChartParser,
    _compare_pair,
    _select_stratified_pairs,
)


def test_canonical_drug_pair_uses_case_insensitive_order():
    assert _canonical_drug_pair("Floxuridine", "bendamustine hydrochloride") == (
        "bendamustine hydrochloride",
        "Floxuridine",
    )


def _write_almanac_archive(path, rows):
    source = path.with_suffix(".csv")
    defaults = {
        "CONCINDEX1": "1",
        "CONCINDEX2": "1",
        "SCREENER": "FG",
        "STUDY": "TEST_STUDY",
    }
    pd.DataFrame([{**defaults, **row} for row in rows]).to_csv(source, index=False)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(source, arcname="ComboDrugGrowth_Nov2017.csv")
    source.unlink()


def test_build_nci_almanac_filters_controls_and_emits_unvalidated_crosswalks(tmp_path):
    archive_path = tmp_path / "almanac.zip"
    _write_almanac_archive(
        archive_path,
        [
            {
                "CELLNAME": "CCRF-CEM",
                "NSC1": "100",
                "NSC2": "200",
                "VALID": "Y",
                "PERCENTGROWTH": "80",
                "EXPECTEDGROWTH": "100",
                "SCORE": "20",
            },
            {
                "CELLNAME": "CCRF-CEM",
                "NSC1": "200",
                "NSC2": "100",
                "VALID": "Y",
                "PERCENTGROWTH": "70",
                "EXPECTEDGROWTH": "100",
                "SCORE": "30",
            },
            {
                "CELLNAME": "A549/ATCC",
                "NSC1": "100",
                "NSC2": "200",
                "VALID": "Y",
                "PERCENTGROWTH": "60",
                "EXPECTEDGROWTH": "100",
                "SCORE": "40",
            },
            {
                "CELLNAME": "A549/ATCC",
                "NSC1": "100",
                "NSC2": "",
                "VALID": "Y",
                "PERCENTGROWTH": "75",
                "EXPECTEDGROWTH": "",
                "SCORE": "",
            },
            {
                "CELLNAME": "CCRF-CEM",
                "NSC1": "100",
                "NSC2": "300",
                "VALID": "N",
                "PERCENTGROWTH": "75",
                "EXPECTEDGROWTH": "100",
                "SCORE": "25",
            },
            {
                "CELLNAME": "CCRF-CEM",
                "NSC1": "752",
                "NSC2": "301",
                "VALID": "Y",
                "PERCENTGROWTH": "70",
                "EXPECTEDGROWTH": "100",
                "SCORE": "30",
            },
        ],
    )
    names_path = tmp_path / "compound_names.txt"
    names_path.write_text(
        "100\tDrug A\n200\tDrug B\n752\tThioguanine\n752\t6-Thioguanine\n"
        "301\t6-thioguanine\n"
    )
    gdsc_path = tmp_path / "gdsc2.csv"
    pd.DataFrame(
        {
            "cell_line": ["CCRF-CEM", "A549"],
            "sanger_model_id": ["SIDM001", "SIDM002"],
            "reported_cancer_type": ["ALL", "LUAD"],
            "drug_id": ["10", "11"],
            "drug_name": ["Drug A", "Drug B"],
        }
    ).to_csv(gdsc_path, index=False)

    outputs = build_nci_almanac_datasets(
        raw_path=archive_path,
        compound_names_path=names_path,
        gdsc_path=gdsc_path,
        output_dir=tmp_path / "processed",
    )

    scores = pd.read_csv(outputs["combo_scores"])
    crosswalk = pd.read_csv(outputs["cell_line_crosswalk"])
    manifest = json.loads(outputs["manifest"].read_text())

    ccrf_score = scores.loc[scores["cell_line"].eq("CCRF-CEM")].iloc[0]
    assert scores[["cell_line", "drug_a", "drug_b"]].values.tolist() == [
        ["A549/ATCC", "Drug A", "Drug B"],
        ["CCRF-CEM", "Drug A", "Drug B"],
    ]
    assert ccrf_score["combo_score"] == 225
    assert ccrf_score["unrounded_difference_sum"] == 50
    assert ccrf_score["dose_row_count"] == 2
    assert manifest["single_agent_or_incomplete_identity_rows_excluded"] == 1
    assert manifest["same_active_drug_pair_rows_excluded"] == 1
    assert manifest["distinct_active_drugs"] == 3
    assert manifest["distinct_unordered_drug_pairs"] == 1
    assert manifest["training_ready"] is False
    assert manifest["dose_reversal_qc_applied"] is True
    assert manifest["cell_line_drug_pair_groups_flagged_for_reversal"] == 0

    exact = crosswalk.loc[crosswalk["almanac_cell_line"].eq("CCRF-CEM")].iloc[0]
    alias = crosswalk.loc[crosswalk["almanac_cell_line"].eq("A549/ATCC")].iloc[0]
    assert exact["candidate_sanger_model_id"] == "SIDM001"
    assert exact["mapping_status"] == "candidate_unvalidated"
    assert alias["candidate_sanger_model_id"] == "SIDM002"
    assert alias["match_method"] == "atcc_suffix_candidate"
    assert alias["mapping_status"] == "candidate_unvalidated"


def test_build_nci_almanac_marks_incomplete_score_groups_unusable(tmp_path):
    archive_path = tmp_path / "almanac.zip"
    _write_almanac_archive(
        archive_path,
        [
            {
                "CELLNAME": "CCRF-CEM",
                "NSC1": "100",
                "NSC2": "200",
                "VALID": "Y",
                "PERCENTGROWTH": "80",
                "EXPECTEDGROWTH": "100",
                "SCORE": "20",
            },
            {
                "CELLNAME": "CCRF-CEM",
                "NSC1": "100",
                "NSC2": "200",
                "VALID": "Y",
                "PERCENTGROWTH": "90",
                "EXPECTEDGROWTH": "",
                "SCORE": "",
            },
            {
                "CELLNAME": "A549",
                "NSC1": "100",
                "NSC2": "200",
                "VALID": "Y",
                "PERCENTGROWTH": "80",
                "EXPECTEDGROWTH": "100",
                "SCORE": "",
            },
            {
                "CELLNAME": "MCF7",
                "NSC1": "100",
                "NSC2": "200",
                "VALID": "Y",
                "PERCENTGROWTH": "75",
                "EXPECTEDGROWTH": "",
                "SCORE": "25",
            },
        ],
    )
    names_path = tmp_path / "compound_names.txt"
    names_path.write_text("100\tDrug A\n200\tDrug B\n")
    gdsc_path = tmp_path / "gdsc2.csv"
    pd.DataFrame(
        {
            "cell_line": ["CCRF-CEM"],
            "sanger_model_id": ["SIDM001"],
            "reported_cancer_type": ["ALL"],
            "drug_id": ["10"],
            "drug_name": ["Drug A"],
        }
    ).to_csv(gdsc_path, index=False)

    outputs = build_nci_almanac_datasets(
        raw_path=archive_path,
        compound_names_path=names_path,
        gdsc_path=gdsc_path,
        output_dir=tmp_path / "processed",
    )

    scores = pd.read_csv(outputs["combo_scores"])
    ccrf_score = scores.loc[scores["cell_line"].eq("CCRF-CEM")].iloc[0]
    score_without_source_value = scores.loc[scores["cell_line"].eq("A549")].iloc[0]
    formula_without_source_value = scores.loc[scores["cell_line"].eq("MCF7")].iloc[0]
    assert ccrf_score["score_complete"] == False
    assert pd.isna(ccrf_score["combo_score"])
    assert ccrf_score["missing_score_row_count"] == 1
    assert not score_without_source_value["score_complete"]
    assert pd.isna(score_without_source_value["combo_score"])
    assert score_without_source_value["formula_complete"]
    assert score_without_source_value["unrounded_difference_sum"] == 20
    assert formula_without_source_value["score_complete"]
    assert formula_without_source_value["combo_score"] == 225
    assert not formula_without_source_value["formula_complete"]
    assert pd.isna(formula_without_source_value["unrounded_difference_sum"])
    manifest = json.loads(outputs["manifest"].read_text())
    assert manifest["score_incomplete_cell_line_drug_pair_rows"] == 2
    assert manifest["formula_incomplete_cell_line_drug_pair_rows"] == 2


def test_build_nci_almanac_excludes_pair_groups_with_adjacent_dose_reversals(tmp_path):
    archive_path = tmp_path / "almanac.zip"
    _write_almanac_archive(
        archive_path,
        [
            {
                "CELLNAME": "CCRF-CEM",
                "NSC1": "100",
                "NSC2": "200",
                "CONCINDEX1": "1",
                "CONCINDEX2": "1",
                "VALID": "Y",
                "PERCENTGROWTH": "20",
                "EXPECTEDGROWTH": "100",
                "SCORE": "80",
            },
            {
                "CELLNAME": "CCRF-CEM",
                "NSC1": "100",
                "NSC2": "200",
                "CONCINDEX1": "2",
                "CONCINDEX2": "1",
                "VALID": "Y",
                "PERCENTGROWTH": "70",
                "EXPECTEDGROWTH": "100",
                "SCORE": "30",
            },
            {
                "CELLNAME": "CCRF-CEM",
                "NSC1": "100",
                "NSC2": "200",
                "CONCINDEX1": "2",
                "CONCINDEX2": "2",
                "VALID": "Y",
                "PERCENTGROWTH": "120",
                "EXPECTEDGROWTH": "100",
                "SCORE": "-20",
            },
            {
                "CELLNAME": "A549",
                "NSC1": "100",
                "NSC2": "200",
                "CONCINDEX1": "1",
                "CONCINDEX2": "1",
                "VALID": "Y",
                "PERCENTGROWTH": "80",
                "EXPECTEDGROWTH": "100",
                "SCORE": "20",
            },
            {
                "CELLNAME": "A549",
                "NSC1": "100",
                "NSC2": "200",
                "CONCINDEX1": "2",
                "CONCINDEX2": "1",
                "VALID": "Y",
                "PERCENTGROWTH": "80",
                "EXPECTEDGROWTH": "100",
                "SCORE": "20",
            },
        ],
    )
    names_path = tmp_path / "compound_names.txt"
    names_path.write_text("100\tDrug A\n200\tDrug B\n")
    gdsc_path = tmp_path / "gdsc2.csv"
    pd.DataFrame(
        {
            "cell_line": ["CCRF-CEM", "A549"],
            "sanger_model_id": ["SIDM001", "SIDM002"],
            "reported_cancer_type": ["ALL", "LUAD"],
            "drug_id": ["10", "11"],
            "drug_name": ["Drug A", "Drug B"],
        }
    ).to_csv(gdsc_path, index=False)

    outputs = build_nci_almanac_datasets(
        raw_path=archive_path,
        compound_names_path=names_path,
        gdsc_path=gdsc_path,
        output_dir=tmp_path / "processed",
    )

    scores = pd.read_csv(outputs["combo_scores"])
    manifest = json.loads(outputs["manifest"].read_text())

    assert scores["cell_line"].tolist() == ["A549"]
    assert manifest["dose_reversal_qc_applied"] is True
    assert manifest["dose_reversal_comparisons_triggered"] == 2
    assert manifest["cell_line_drug_pair_groups_flagged_for_reversal"] == 1
    assert manifest["cell_line_drug_pair_groups_excluded_for_reversal"] == 1
    exclusions = pd.read_csv(outputs["reversal_exclusions"])
    assert len(exclusions) == 1
    assert exclusions.iloc[0]["cell_line"] == "CCRF-CEM"


def test_study_combo_score_matches_nine_point_and_normalized_fifteen_point_grids():
    assert _normalize_study_combo_score(114, 15) == 68
    assert _normalize_study_combo_score(-28, 15) == -16
    assert _normalize_study_combo_score(30, 9) == 30


def test_gdsc_drug_crosswalk_keeps_exact_name_matches_as_candidates(tmp_path):
    gdsc_path = tmp_path / "gdsc2.csv"
    pd.DataFrame(
        {
            "drug_id": ["GDSC001", "GDSC002", "GDSC003"],
            "drug_name": ["Drug A", "drug a", "Drug B"],
        }
    ).to_csv(gdsc_path, index=False)

    crosswalk = _gdsc_drug_crosswalk({"DRUG A", "Drug B", "Drug C"}, gdsc_path)
    ambiguous = crosswalk.loc[
        crosswalk["almanac_drug_name"].eq("DRUG A")
    ].iloc[0]
    unique = crosswalk.loc[crosswalk["almanac_drug_name"].eq("Drug B")].iloc[0]
    unmatched = crosswalk.loc[crosswalk["almanac_drug_name"].eq("Drug C")].iloc[0]

    assert ambiguous["candidate_gdsc_drug_ids"] == "GDSC001|GDSC002"
    assert ambiguous["mapping_status"] == "ambiguous"
    assert unique["candidate_gdsc_drug_ids"] == "GDSC003"
    assert unique["mapping_status"] == "candidate_unvalidated"
    assert unmatched["mapping_status"] == "unmatched"


def test_official_chart_parser_extracts_study_level_scores():
    parser = _ScoreChartParser()
    parser.feed(
        """
        <map>
          <area title="(102816-719344, A549/ATCC [LNS][FF]) = -53"
            href="getDoseResponseGraphs.do?nsc1=102816&amp;nsc2=719344&amp;study=CD025CP16"/>
          <area title="Drug pair without a score" href="getScoresChart.do"/>
        </map>
        """
    )

    assert parser.records == [
        {
            "cell_line": "A549/ATCC",
            "screener": "FF",
            "study": "CD025CP16",
            "official_score": -53,
        }
    ]


def test_official_validation_sample_is_stratified_and_repeatable():
    rows = []
    nsc_pairs = {}
    for index in range(10):
        drug_a = f"Drug {index:02d}A"
        drug_b = f"Drug {index:02d}B"
        pair = tuple(sorted((drug_a, drug_b), key=str.casefold))
        nsc_pairs[pair] = {(str(1000 + index * 2), str(1001 + index * 2))}
        rows.extend(
            [
                {
                    "cell_line": "Cell A",
                    "drug_a": drug_a,
                    "drug_b": drug_b,
                    "combo_score": index * 10,
                    "score_complete": True,
                },
                {
                    "cell_line": "Cell B",
                    "drug_a": drug_a,
                    "drug_b": drug_b,
                    "combo_score": index * 10 + 2,
                    "score_complete": True,
                },
            ]
        )
    scores = pd.DataFrame(rows)

    first = _select_stratified_pairs(scores, nsc_pairs, sample_size=10, seed=7)
    second = _select_stratified_pairs(scores, nsc_pairs, sample_size=10, seed=7)

    assert first.equals(second)
    assert first["score_stratum"].value_counts().sort_index().tolist() == [2] * 5


def test_official_validation_compares_cell_line_score_sums():
    sample = {
        "drug_a": "Drug A",
        "drug_b": "Drug B",
        "nsc1": "100",
        "nsc2": "200",
        "score_stratum": 2,
    }
    local = pd.DataFrame(
        [
            {"cell_line": "Cell A", "drug_a": "Drug A", "drug_b": "Drug B", "combo_score": -5},
            {"cell_line": "Cell B", "drug_a": "Drug A", "drug_b": "Drug B", "combo_score": 8},
        ]
    )
    official = [
        {"cell_line": "Cell A", "study": "S1", "official_score": -2},
        {"cell_line": "Cell A", "study": "S2", "official_score": -3},
        {"cell_line": "Cell B", "study": "S1", "official_score": 7},
    ]

    summary, details = _compare_pair(sample, local, official)

    assert summary["exact_matches"] == 1
    assert summary["score_mismatches"] == 1
    assert [(row["cell_line"], row["status"]) for row in details] == [
        ("Cell A", "exact_match"),
        ("Cell B", "score_mismatch"),
    ]


def test_official_validation_marks_exact_reversal_exclusions():
    sample = {
        "drug_a": "Drug A",
        "drug_b": "Drug B",
        "nsc1": "100",
        "nsc2": "200",
        "score_stratum": 1,
    }
    local = pd.DataFrame(
        [
            {
                "cell_line": "Cell A",
                "drug_a": "Drug A",
                "drug_b": "Drug B",
                "combo_score": 4,
                "reversal_qc_excluded": True,
            }
        ]
    )
    official = [{"cell_line": "Cell A", "study": "S1", "official_score": 4}]

    summary, details = _compare_pair(sample, local, official)

    assert summary["exact_matches"] == 0
    assert summary["exact_matches_reversal_excluded"] == 1
    assert details[0]["status"] == "exact_match_reversal_excluded"
