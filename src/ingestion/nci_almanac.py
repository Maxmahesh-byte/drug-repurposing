from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import CONFIG

ALMANAC_ARCHIVE_FILENAME = "ComboDrugGrowth_Nov2017.zip"
ALMANAC_COMPOUND_NAMES_FILENAME = "ComboCompoundNames_small.txt"
ALMANAC_COLUMNS = {
    "CELLNAME",
    "NSC1",
    "NSC2",
    "VALID",
    "CONCINDEX1",
    "CONCINDEX2",
    "PERCENTGROWTH",
    "EXPECTEDGROWTH",
    "SCORE",
    "SCREENER",
    "STUDY",
}
DOSE_REVERSAL_THRESHOLD = 50.0
DRUG_NAME_ALIASES = {
    "6-thioguanine": "thioguanine",
    "plicamycin": "mithramycin",
    "vinblastine sulfate hydrate": "vinblastine sulfate",
}
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]+")
_OUTPUT_COLUMNS = [
    "cell_line",
    "drug_a",
    "drug_b",
    "combo_score",
    "unrounded_difference_sum",
    "dose_row_count",
    "scored_dose_row_count",
    "missing_score_row_count",
    "score_complete",
    "missing_formula_row_count",
    "formula_complete",
]


def _canonical_drug_pair(drug_a: str, drug_b: str) -> tuple[str, str]:
    return tuple(sorted((drug_a, drug_b), key=str.casefold))


def _resolve_source(path: str | Path | None, env_var: str, filename: str) -> Path:
    configured = os.environ.get(env_var)
    if configured:
        resolved = Path(configured).expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(f"{env_var} does not point to a file: {resolved}")
        return resolved
    if path is not None:
        resolved = Path(path).expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(f"ALMANAC source file does not exist: {resolved}")
        return resolved
    resolved = Path(CONFIG["raw_data_path"]) / "almanac" / filename
    if not resolved.is_file():
        raise FileNotFoundError(
            f"ALMANAC source file not found: {resolved}; set {env_var} or pass its path"
        )
    return resolved


def _read_chunks(source_path: Path, chunksize: int = 100_000) -> Iterator[pd.DataFrame]:
    if zipfile.is_zipfile(source_path):
        with zipfile.ZipFile(source_path) as archive:
            csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(csv_members) != 1:
                raise ValueError(
                    f"Expected one CSV file in the ALMANAC archive; found {len(csv_members)}"
                )
            with archive.open(csv_members[0]) as source:
                yield from pd.read_csv(
                    source,
                    usecols=lambda column: column in ALMANAC_COLUMNS,
                    dtype=str,
                    keep_default_na=False,
                    chunksize=chunksize,
                )
        return

    yield from pd.read_csv(
        source_path,
        usecols=lambda column: column in ALMANAC_COLUMNS,
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    )


def _load_compound_map(path: Path) -> tuple[dict[str, str], pd.DataFrame]:
    names = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["nsc", "source_name"],
        dtype=str,
        keep_default_na=False,
    )
    names["nsc"] = names["nsc"].str.strip()
    names["source_name"] = names["source_name"].str.strip()
    names = names[names["nsc"].ne("") & names["source_name"].ne("")].drop_duplicates()
    names["canonical_drug_name"] = (
        names["source_name"].str.casefold().replace(DRUG_NAME_ALIASES)
    )
    ambiguous = names.groupby("nsc")["canonical_drug_name"].nunique()
    ambiguous = ambiguous[ambiguous.gt(1)]
    if not ambiguous.empty:
        raise ValueError(
            "ALMANAC NSC identifiers map to multiple drug identities: "
            f"{ambiguous.index.tolist()[:10]}"
        )

    canonical_display = (
        names.assign(
            _display_length=names["source_name"].str.len(),
            _display_folded=names["source_name"].str.casefold(),
        )
        .sort_values(["_display_length", "_display_folded", "source_name"])
        .drop_duplicates("canonical_drug_name")
        .set_index("canonical_drug_name")["source_name"]
        .to_dict()
    )
    names["canonical_drug_name"] = names["canonical_drug_name"].map(canonical_display)
    names["source_names"] = names["nsc"].map(
        names.groupby("nsc")["source_name"].agg(lambda values: " | ".join(sorted(set(values))))
    )
    nsc_to_name = names.drop_duplicates("nsc").set_index("nsc")["canonical_drug_name"].to_dict()
    drug_crosswalk = (
        names.drop_duplicates("nsc")
        .loc[:, ["nsc", "source_names", "canonical_drug_name"]]
        .sort_values("nsc", key=lambda values: pd.to_numeric(values))
        .reset_index(drop=True)
    )
    return nsc_to_name, drug_crosswalk


def _clean_cell_name(values: pd.Series) -> pd.Series:
    return values.astype(str).map(lambda value: _CONTROL_CHARACTERS.sub("", value)).str.strip()


def _cell_line_crosswalk(
    almanac_cell_lines: set[str],
    gdsc_path: Path,
) -> pd.DataFrame:
    if not gdsc_path.is_file():
        raise FileNotFoundError(f"GDSC2 response file is required for candidate matching: {gdsc_path}")

    gdsc = pd.read_csv(
        gdsc_path,
        usecols=["cell_line", "sanger_model_id", "reported_cancer_type"],
        dtype=str,
        keep_default_na=False,
    )
    gdsc["cell_line"] = gdsc["cell_line"].map(
        lambda value: _CONTROL_CHARACTERS.sub("", value).strip()
    )
    gdsc["sanger_model_id"] = gdsc["sanger_model_id"].str.strip()
    gdsc["reported_cancer_type"] = gdsc["reported_cancer_type"].str.strip()
    if gdsc[["cell_line", "sanger_model_id"]].eq("").any().any():
        raise ValueError("GDSC2 cell-line data contain empty names or Sanger model IDs")
    if gdsc.groupby("cell_line")["sanger_model_id"].nunique().gt(1).any():
        raise ValueError("A GDSC2 cell-line name maps to multiple Sanger model IDs")

    exact: dict[str, list[dict[str, str]]] = {}
    atcc_normalized: dict[str, list[dict[str, str]]] = {}
    for row in gdsc.drop_duplicates(["cell_line", "sanger_model_id"]).to_dict("records"):
        name = row["cell_line"]
        exact.setdefault(name.casefold(), []).append(row)
        normalized = re.sub(r"/atcc$", "", name, flags=re.IGNORECASE).casefold()
        atcc_normalized.setdefault(normalized, []).append(row)

    rows = []
    for almanac_name in sorted(almanac_cell_lines, key=str.casefold):
        exact_matches = exact.get(almanac_name.casefold(), [])
        if exact_matches:
            matches = exact_matches
            match_method = "case_insensitive_exact_name"
        else:
            normalized = re.sub(r"/atcc$", "", almanac_name, flags=re.IGNORECASE).casefold()
            matches = atcc_normalized.get(normalized, [])
            match_method = "atcc_suffix_candidate"

        unique_matches = {
            (row["cell_line"], row["sanger_model_id"], row["reported_cancer_type"])
            for row in matches
        }
        if len(unique_matches) == 1:
            cell_line, model_id, cancer_type = next(iter(unique_matches))
            status = "candidate_unvalidated"
            note = "Name-based candidate only; confirm stable cell-line identity before joining features."
        elif len(unique_matches) > 1:
            cell_line = model_id = cancer_type = ""
            status = "ambiguous"
            note = "Normalized name matches multiple GDSC2 models; do not merge automatically."
        else:
            cell_line = model_id = cancer_type = ""
            match_method = "none"
            status = "unmatched"
            note = "No GDSC2 name candidate found."
        rows.append(
            {
                "almanac_cell_line": almanac_name,
                "candidate_gdsc_cell_line": cell_line,
                "candidate_sanger_model_id": model_id,
                "candidate_reported_cancer_type": cancer_type,
                "match_method": match_method,
                "mapping_status": status,
                "review_note": note,
            }
        )
    return pd.DataFrame(rows)


def _gdsc_drug_crosswalk(
    almanac_drug_names: set[str],
    gdsc_path: Path,
) -> pd.DataFrame:
    if not gdsc_path.is_file():
        raise FileNotFoundError(f"GDSC2 response file is required for drug candidates: {gdsc_path}")

    gdsc = pd.read_csv(
        gdsc_path,
        usecols=["drug_id", "drug_name"],
        dtype=str,
        keep_default_na=False,
    )
    gdsc["drug_id"] = gdsc["drug_id"].str.strip()
    gdsc["drug_name"] = gdsc["drug_name"].str.strip()
    if gdsc[["drug_id", "drug_name"]].eq("").any().any():
        raise ValueError("GDSC2 drug data contain empty drug names or IDs")

    candidates: dict[str, dict[str, set[str]]] = {}
    for row in gdsc.drop_duplicates(["drug_id", "drug_name"]).to_dict("records"):
        key = row["drug_name"].casefold()
        entry = candidates.setdefault(key, {})
        entry.setdefault(row["drug_id"], set()).add(row["drug_name"])

    rows = []
    for almanac_name in sorted(almanac_drug_names, key=str.casefold):
        matches = candidates.get(almanac_name.casefold(), {})
        drug_ids = sorted(matches)
        candidate_names = sorted(
            {name for names in matches.values() for name in names},
            key=str.casefold,
        )
        if len(drug_ids) == 1:
            status = "candidate_unvalidated"
            note = "Case-insensitive exact-name candidate only; confirm chemical identity."
        elif len(drug_ids) > 1:
            status = "ambiguous"
            note = "Exact name maps to multiple GDSC2 drug IDs; do not merge automatically."
        else:
            status = "unmatched"
            note = "No case-insensitive exact GDSC2 drug-name candidate found."
        rows.append(
            {
                "almanac_drug_name": almanac_name,
                "candidate_gdsc_drug_ids": "|".join(drug_ids),
                "candidate_gdsc_drug_names": "|".join(candidate_names),
                "match_method": (
                    "case_insensitive_exact_name" if drug_ids else "none"
                ),
                "mapping_status": status,
                "review_note": note,
            }
        )
    return pd.DataFrame(rows)


def _create_reversal_table(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute(
        """
        CREATE TABLE dose_points (
            cell_line TEXT NOT NULL,
            drug_a TEXT NOT NULL,
            drug_b TEXT NOT NULL,
            screener TEXT NOT NULL,
            study TEXT NOT NULL,
            dose_index_a INTEGER NOT NULL,
            dose_index_b INTEGER NOT NULL,
            growth_sum REAL NOT NULL,
            row_count INTEGER NOT NULL,
            score_sum REAL NOT NULL,
            score_row_count INTEGER NOT NULL,
            PRIMARY KEY (
                cell_line, drug_a, drug_b, screener, study, dose_index_a, dose_index_b
            )
        ) WITHOUT ROWID
        """
    )


def _record_dose_points(
    connection: sqlite3.Connection,
    rows: pd.DataFrame,
) -> None:
    points = (
        rows.groupby(
            [
                "CELLNAME",
                "drug_a",
                "drug_b",
                "SCREENER",
                "STUDY",
                "_dose_index_a",
                "_dose_index_b",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            growth_sum=("_growth", "sum"),
            row_count=("_growth", "size"),
            score_sum=("_source_score", "sum"),
            score_row_count=("_source_score", "count"),
        )
    )
    connection.executemany(
        """
        INSERT INTO dose_points VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (
            cell_line, drug_a, drug_b, screener, study, dose_index_a, dose_index_b
        ) DO UPDATE SET
            growth_sum = growth_sum + excluded.growth_sum,
            row_count = row_count + excluded.row_count,
            score_sum = score_sum + excluded.score_sum,
            score_row_count = score_row_count + excluded.score_row_count
        """,
        points.itertuples(index=False, name=None),
    )
    connection.commit()


def _find_reversal_groups(
    connection: sqlite3.Connection,
) -> tuple[dict[tuple[str, str, str], int], int]:
    comparisons = connection.execute(
        """
        WITH point_means AS (
            SELECT
                cell_line,
                drug_a,
                drug_b,
                screener,
                study,
                dose_index_a,
                dose_index_b,
                growth_sum / row_count AS mean_growth
            FROM dose_points
        ),
        reversals AS (
            SELECT
                lower.cell_line,
                lower.drug_a,
                lower.drug_b,
                lower.screener,
                lower.study,
                'drug_a' AS dose_axis,
                lower.dose_index_a AS lower_dose_index
            FROM point_means AS lower
            JOIN point_means AS higher
              ON higher.cell_line = lower.cell_line
             AND higher.drug_a = lower.drug_a
             AND higher.drug_b = lower.drug_b
             AND higher.screener = lower.screener
             AND higher.study = lower.study
             AND higher.dose_index_a = lower.dose_index_a + 1
             AND higher.dose_index_b = lower.dose_index_b
            WHERE higher.mean_growth - lower.mean_growth >= ?

            UNION ALL

            SELECT
                lower.cell_line,
                lower.drug_a,
                lower.drug_b,
                lower.screener,
                lower.study,
                'drug_b' AS dose_axis,
                lower.dose_index_b AS lower_dose_index
            FROM point_means AS lower
            JOIN point_means AS higher
              ON higher.cell_line = lower.cell_line
             AND higher.drug_a = lower.drug_a
             AND higher.drug_b = lower.drug_b
             AND higher.screener = lower.screener
             AND higher.study = lower.study
             AND higher.dose_index_a = lower.dose_index_a
             AND higher.dose_index_b = lower.dose_index_b + 1
            WHERE higher.mean_growth - lower.mean_growth >= ?
        )
        SELECT cell_line, drug_a, drug_b, COUNT(*)
        FROM reversals
        GROUP BY cell_line, drug_a, drug_b
        """,
        (DOSE_REVERSAL_THRESHOLD, DOSE_REVERSAL_THRESHOLD),
    ).fetchall()
    groups = {
        (cell_line, drug_a, drug_b): count
        for cell_line, drug_a, drug_b, count in comparisons
    }
    comparison_count = sum(groups.values())
    return groups, comparison_count


def _study_dose_grid_sizes(connection: sqlite3.Connection) -> dict[tuple[str, str, str, str], int]:
    rows = connection.execute(
        """
        SELECT drug_a, drug_b, screener, study, COUNT(*)
        FROM (
            SELECT DISTINCT
                drug_a, drug_b, screener, study, dose_index_a, dose_index_b
            FROM dose_points
        )
        GROUP BY drug_a, drug_b, screener, study
        """
    ).fetchall()
    return {
        (drug_a, drug_b, screener, study): point_count
        for drug_a, drug_b, screener, study, point_count in rows
    }


def _study_point_score_sums(connection: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT
            cell_line,
            drug_a,
            drug_b,
            screener AS SCREENER,
            study AS STUDY,
            SUM(
                CASE
                    WHEN score_row_count > 0
                    THEN score_sum / score_row_count
                    ELSE 0
                END
            ) AS point_score_sum
        FROM dose_points
        GROUP BY cell_line, drug_a, drug_b, screener, study
        """,
        connection,
    )


def _normalize_study_combo_score(score_sum: float, dose_grid_size: int) -> int:
    if dose_grid_size <= 0:
        raise ValueError(f"Invalid ALMANAC dose-grid size: {dose_grid_size}")
    return math.trunc(score_sum * 9 / dose_grid_size)


def _build_nci_almanac_datasets(
    reversal_db: sqlite3.Connection,
    raw_path: str | Path | None = None,
    compound_names_path: str | Path | None = None,
    gdsc_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Create reversal-QC-filtered scores and candidate cell-line crosswalks."""
    raw_source = _resolve_source(raw_path, "NCI_ALMANAC_PATH", ALMANAC_ARCHIVE_FILENAME)
    names_source = _resolve_source(
        compound_names_path,
        "NCI_ALMANAC_COMPOUND_NAMES_PATH",
        ALMANAC_COMPOUND_NAMES_FILENAME,
    )
    response_source = (
        Path(gdsc_path).expanduser()
        if gdsc_path is not None
        else Path(CONFIG["raw_data_path"]) / "gdsc" / "gdsc2_responses.csv"
    )
    output_root = (
        Path(output_dir)
        if output_dir is not None
        else Path(CONFIG["processed_data_path"]) / "almanac"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    nsc_to_name, drug_crosswalk = _load_compound_map(names_source)

    _create_reversal_table(reversal_db)
    rows_read = valid_rows = single_agent_rows = combo_rows = 0
    self_pair_rows = 0
    unmapped_nscs: set[str] = set()
    almanac_cell_lines: set[str] = set()
    partials: list[pd.DataFrame] = []
    score_rounding_mismatches = 0

    for chunk in _read_chunks(raw_source):
        absent = ALMANAC_COLUMNS - set(chunk.columns)
        if absent:
            raise ValueError(f"ALMANAC source is missing required columns: {sorted(absent)}")
        rows_read += len(chunk)
        chunk["CELLNAME"] = _clean_cell_name(chunk["CELLNAME"])
        chunk["NSC1"] = chunk["NSC1"].str.strip()
        chunk["NSC2"] = chunk["NSC2"].str.strip()
        valid = chunk["VALID"].str.strip().str.upper().eq("Y")
        valid_chunk = chunk.loc[valid].copy()
        valid_rows += len(valid_chunk)
        combination = valid_chunk["NSC1"].ne("") & valid_chunk["NSC2"].ne("")
        single_agent_rows += int((~combination).sum())
        valid_chunk = valid_chunk.loc[combination].copy()
        combo_rows += len(valid_chunk)
        if valid_chunk.empty:
            continue

        nsc1 = valid_chunk["NSC1"].map(nsc_to_name)
        nsc2 = valid_chunk["NSC2"].map(nsc_to_name)
        missing_ids = set(valid_chunk.loc[nsc1.isna(), "NSC1"]) | set(
            valid_chunk.loc[nsc2.isna(), "NSC2"]
        )
        if missing_ids:
            unmapped_nscs.update(missing_ids)
            continue
        almanac_cell_lines.update(valid_chunk["CELLNAME"].loc[valid_chunk["CELLNAME"].ne("")])
        same_drug = nsc1.eq(nsc2)
        self_pair_rows += int(same_drug.sum())
        valid_chunk = valid_chunk.loc[~same_drug].copy()
        nsc1 = nsc1.loc[valid_chunk.index]
        nsc2 = nsc2.loc[valid_chunk.index]
        if valid_chunk.empty:
            continue

        forward = nsc1.str.casefold().le(nsc2.str.casefold())
        valid_chunk["drug_a"] = nsc1.where(forward, nsc2)
        valid_chunk["drug_b"] = nsc2.where(forward, nsc1)
        dose_index_1 = pd.to_numeric(valid_chunk["CONCINDEX1"], errors="coerce")
        dose_index_2 = pd.to_numeric(valid_chunk["CONCINDEX2"], errors="coerce")
        growth = pd.to_numeric(valid_chunk["PERCENTGROWTH"], errors="coerce")
        if (
            dose_index_1.isna().any()
            or dose_index_2.isna().any()
            or dose_index_1.le(0).any()
            or dose_index_2.le(0).any()
            or dose_index_1.mod(1).ne(0).any()
            or dose_index_2.mod(1).ne(0).any()
            or growth.isna().any()
            or valid_chunk[["SCREENER", "STUDY", "CELLNAME"]].eq("").any().any()
        ):
            raise ValueError(
                "Combination rows contain missing growth, dose-index, cell-line, "
                "or assay provenance needed for reversal QC"
            )
        valid_chunk["_dose_index_a"] = (
            dose_index_1.where(forward, dose_index_2).astype("int64")
        )
        valid_chunk["_dose_index_b"] = (
            dose_index_2.where(forward, dose_index_1).astype("int64")
        )
        valid_chunk["_growth"] = growth
        observed = pd.to_numeric(valid_chunk["PERCENTGROWTH"], errors="coerce")
        expected = pd.to_numeric(valid_chunk["EXPECTEDGROWTH"], errors="coerce")
        source_score = pd.to_numeric(valid_chunk["SCORE"], errors="coerce")
        calculated_score = expected - observed
        comparable = calculated_score.notna() & source_score.notna()
        score_rounding_mismatches += int(
            (calculated_score.loc[comparable].round().ne(source_score.loc[comparable])).sum()
        )
        valid_chunk["_calculated_score"] = calculated_score
        valid_chunk["_source_score"] = source_score
        valid_chunk["_score_present"] = source_score.notna().astype("int64")
        valid_chunk["_formula_present"] = calculated_score.notna().astype("int64")
        _record_dose_points(reversal_db, valid_chunk)
        valid_chunk["_rounding_mismatch"] = (
            comparable & calculated_score.round().ne(source_score)
        ).astype("int64")
        valid_chunk["cell_line"] = valid_chunk["CELLNAME"]
        assay_columns = [
            "cell_line",
            "drug_a",
            "drug_b",
            "SCREENER",
            "STUDY",
        ]
        grouped = (
            valid_chunk.groupby(
                assay_columns,
                as_index=False,
                sort=False,
                dropna=False,
            )
            .agg(
                dose_row_count=("NSC1", "size"),
                scored_dose_row_count=("_score_present", "sum"),
                formula_dose_row_count=("_formula_present", "sum"),
                calculated_score_sum=("_calculated_score", "sum"),
                source_score_sum=("_source_score", "sum"),
                rounding_mismatch_count=("_rounding_mismatch", "sum"),
            )
        )
        partials.append(grouped)

    if unmapped_nscs:
        raise ValueError(
            "ALMANAC NSC identifiers are absent from the source compound-name map: "
            f"{sorted(unmapped_nscs)}"
        )
    if not partials:
        raise ValueError("No valid, distinct-drug ALMANAC combination records were found")

    scores = (
        pd.concat(partials, ignore_index=True)
        .groupby(
            ["cell_line", "drug_a", "drug_b", "SCREENER", "STUDY"],
            as_index=False,
            sort=False,
        )
        .agg(
            dose_row_count=("dose_row_count", "sum"),
            scored_dose_row_count=("scored_dose_row_count", "sum"),
            formula_dose_row_count=("formula_dose_row_count", "sum"),
            calculated_score_sum=("calculated_score_sum", "sum"),
            source_score_sum=("source_score_sum", "sum"),
            rounding_mismatch_count=("rounding_mismatch_count", "sum"),
        )
    )
    study_grid_sizes = _study_dose_grid_sizes(reversal_db)
    scores["dose_grid_size"] = [
        study_grid_sizes[(drug_a, drug_b, screener, study)]
        for drug_a, drug_b, screener, study in scores[
            ["drug_a", "drug_b", "SCREENER", "STUDY"]
        ].itertuples(index=False, name=None)
    ]
    point_scores = _study_point_score_sums(reversal_db)
    scores = scores.merge(
        point_scores,
        on=["cell_line", "drug_a", "drug_b", "SCREENER", "STUDY"],
        how="left",
        validate="one_to_one",
    )
    scores["study_combo_score"] = [
        _normalize_study_combo_score(score_sum, grid_size)
        for score_sum, grid_size in scores[
            ["point_score_sum", "dose_grid_size"]
        ].itertuples(index=False, name=None)
    ]
    scores = (
        scores.groupby(
            ["cell_line", "drug_a", "drug_b"],
            as_index=False,
            sort=False,
        )
        .agg(
            dose_row_count=("dose_row_count", "sum"),
            scored_dose_row_count=("scored_dose_row_count", "sum"),
            formula_dose_row_count=("formula_dose_row_count", "sum"),
            calculated_score_sum=("calculated_score_sum", "sum"),
            study_combo_score_sum=("study_combo_score", "sum"),
            rounding_mismatch_count=("rounding_mismatch_count", "sum"),
        )
    )
    scores["missing_score_row_count"] = (
        scores["dose_row_count"] - scores["scored_dose_row_count"]
    )
    scores["score_complete"] = scores["missing_score_row_count"].eq(0)
    scores["combo_score"] = scores["study_combo_score_sum"].where(scores["score_complete"])
    scores["missing_formula_row_count"] = (
        scores["dose_row_count"] - scores["formula_dose_row_count"]
    )
    scores["formula_complete"] = scores["missing_formula_row_count"].eq(0)
    scores["unrounded_difference_sum"] = scores["calculated_score_sum"].where(
        scores["formula_complete"]
    )
    reversal_groups, reversal_comparison_count = _find_reversal_groups(reversal_db)
    reversal_mask = pd.Series(
        [
            (cell_line, drug_a, drug_b) in reversal_groups
            for cell_line, drug_a, drug_b in scores[
                ["cell_line", "drug_a", "drug_b"]
            ].itertuples(index=False, name=None)
        ],
        index=scores.index,
    )
    reversal_exclusions = scores.loc[reversal_mask].copy()
    reversal_exclusions["reversal_comparison_count"] = [
        reversal_groups[(cell_line, drug_a, drug_b)]
        for cell_line, drug_a, drug_b in reversal_exclusions[
            ["cell_line", "drug_a", "drug_b"]
        ].itertuples(index=False, name=None)
    ]
    reversal_exclusions = reversal_exclusions[_OUTPUT_COLUMNS + ["reversal_comparison_count"]]
    scores = scores.loc[~reversal_mask].copy()
    scores = scores[_OUTPUT_COLUMNS].sort_values(
        ["cell_line", "drug_a", "drug_b"], key=lambda values: values.str.casefold()
    )
    reversal_exclusions = reversal_exclusions.sort_values(
        ["cell_line", "drug_a", "drug_b"], key=lambda values: values.str.casefold()
    )

    crosswalk = _cell_line_crosswalk(almanac_cell_lines, response_source)
    gdsc_drug_crosswalk = _gdsc_drug_crosswalk(
        set(drug_crosswalk["canonical_drug_name"]),
        response_source,
    )
    output_paths = {
        "combo_scores": output_root / "nci_almanac_combo_scores.csv",
        "drug_crosswalk": output_root / "nci_almanac_drug_crosswalk.csv",
        "gdsc_drug_crosswalk": (
            output_root / "nci_almanac_gdsc_drug_crosswalk_candidates.csv"
        ),
        "cell_line_crosswalk": output_root / "nci_almanac_cell_line_crosswalk_candidates.csv",
        "reversal_exclusions": output_root / "nci_almanac_reversal_exclusions.csv",
        "manifest": output_root / "nci_almanac_manifest.json",
    }
    dataframes = {
        "combo_scores": scores,
        "drug_crosswalk": drug_crosswalk,
        "gdsc_drug_crosswalk": gdsc_drug_crosswalk,
        "cell_line_crosswalk": crosswalk,
        "reversal_exclusions": reversal_exclusions,
    }
    temporary_paths = {}
    try:
        for key, dataframe in dataframes.items():
            temporary = output_paths[key].with_suffix(".csv.tmp")
            dataframe.to_csv(temporary, index=False)
            temporary_paths[key] = temporary
        manifest: dict[str, Any] = {
            "source": "NCI-ALMANAC ComboDrugGrowth_Nov2017",
            "source_archive": str(raw_source),
            "compound_names_source": str(names_source),
            "gdsc2_source": str(response_source),
            "rows_read": rows_read,
            "valid_rows": valid_rows,
            "single_agent_or_incomplete_identity_rows_excluded": single_agent_rows,
            "combination_dose_rows_before_same_drug_filter": combo_rows,
            "same_active_drug_pair_rows_excluded": self_pair_rows,
            "distinct_active_drugs": len(drug_crosswalk["canonical_drug_name"].unique()),
            "distinct_unordered_drug_pairs": int(
                scores[["drug_a", "drug_b"]].drop_duplicates().shape[0]
            ),
            "cell_line_drug_pair_rows": len(scores),
            "score_complete_cell_line_drug_pair_rows": int(scores["score_complete"].sum()),
            "score_incomplete_cell_line_drug_pair_rows": int((~scores["score_complete"]).sum()),
            "scored_dose_rows": int(scores["scored_dose_row_count"].sum()),
            "missing_score_dose_rows": int(scores["missing_score_row_count"].sum()),
            "formula_complete_cell_line_drug_pair_rows": int(scores["formula_complete"].sum()),
            "formula_incomplete_cell_line_drug_pair_rows": int(
                (~scores["formula_complete"]).sum()
            ),
            "formula_scored_dose_rows": int(
                scores["dose_row_count"].sum() - scores["missing_formula_row_count"].sum()
            ),
            "missing_formula_dose_rows": int(scores["missing_formula_row_count"].sum()),
            "source_score_rounding_mismatches": score_rounding_mismatches,
            "score_definition": (
                "For each cell line, unordered drug pair, screener, and study, average "
                "row-level SCORE values at replicate dose points, sum the dose-point "
                "means, normalize to a nine-point equivalent, truncate toward zero to "
                "an integer, then sum study scores. Reversal-flagged cell-line/pair "
                "groups are excluded from combo_scores."
            ),
            "source_score_note": (
                "Study scores use 9 / observed study dose-grid size as the normalization "
                "factor; replicate SCORE rows at the same dose point are averaged first. "
                "The unrounded EXPECTEDGROWTH - PERCENTGROWTH sum is retained as a "
                "separate formula diagnostic."
            ),
            "dose_reversal_qc_applied": True,
            "dose_reversal_threshold_growth_percentage_points": DOSE_REVERSAL_THRESHOLD,
            "dose_reversal_comparisons_triggered": reversal_comparison_count,
            "cell_line_drug_pair_groups_flagged_for_reversal": len(reversal_groups),
            "cell_line_drug_pair_groups_excluded_for_reversal": len(reversal_groups),
            "reversal_exclusion_groups_with_scores": len(reversal_exclusions),
            "independent_combo_score_validation": False,
            "training_ready": False,
            "cell_line_name_candidates": len(crosswalk),
            "cell_line_name_candidate_matches": int(
                crosswalk["mapping_status"].eq("candidate_unvalidated").sum()
            ),
            "cell_line_name_unmatched": int(crosswalk["mapping_status"].eq("unmatched").sum()),
            "cell_line_name_ambiguous": int(crosswalk["mapping_status"].eq("ambiguous").sum()),
            "gdsc_drug_name_candidates": int(
                gdsc_drug_crosswalk["mapping_status"]
                .eq("candidate_unvalidated")
                .sum()
            ),
            "gdsc_drug_name_exact_matches": int(
                gdsc_drug_crosswalk["mapping_status"]
                .isin(["candidate_unvalidated", "ambiguous"])
                .sum()
            ),
            "gdsc_drug_name_unmatched": int(
                gdsc_drug_crosswalk["mapping_status"].eq("unmatched").sum()
            ),
            "gdsc_drug_name_ambiguous": int(
                gdsc_drug_crosswalk["mapping_status"].eq("ambiguous").sum()
            ),
            "cell_line_identity_warning": (
                "GDSC2 matches are name-based candidates only, not validated stable-ID links."
            ),
            "outputs": {
                key: str(path)
                for key, path in output_paths.items()
                if key != "manifest"
            },
        }
        for key, temporary in temporary_paths.items():
            temporary.replace(output_paths[key])
        output_paths["manifest"].write_text(json.dumps(manifest, indent=2))
    finally:
        for temporary in temporary_paths.values():
            if temporary.exists():
                temporary.unlink()
    return output_paths


def build_nci_almanac_datasets(
    raw_path: str | Path | None = None,
    compound_names_path: str | Path | None = None,
    gdsc_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Build ALMANAC outputs using a temporary on-disk dose-reversal index."""
    with tempfile.TemporaryDirectory(prefix="nci_almanac_qc_") as temporary_dir:
        db_path = Path(temporary_dir) / "dose_reversal.sqlite"
        with closing(sqlite3.connect(db_path)) as reversal_db:
            return _build_nci_almanac_datasets(
                raw_path=raw_path,
                compound_names_path=compound_names_path,
                gdsc_path=gdsc_path,
                output_dir=output_dir,
                reversal_db=reversal_db,
            )


if __name__ == "__main__":
    outputs = build_nci_almanac_datasets()
    for name, path in outputs.items():
        print(f"{name}: {path}")
