from __future__ import annotations

import json
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit

import pandas as pd
import requests

from src.ingestion.nci_almanac import (
    ALMANAC_ARCHIVE_FILENAME,
    ALMANAC_COLUMNS,
    ALMANAC_COMPOUND_NAMES_FILENAME,
    _canonical_drug_pair,
    _load_compound_map,
)
from src.utils.config import CONFIG

OFFICIAL_CHART_ENDPOINT = "https://dtp.cancer.gov/ncialmanac/getScoresChart.do"
SAMPLE_SIZE = 100
SAMPLE_SEED = 20261102
SCORE_TITLE_PATTERN = re.compile(
    r"\([^,]+,\s*(.*?)\s+\[[^]]+\]\[([^]]+)\]\)\s*=\s*(-?\d+)\s*$"
)


class _ScoreChartParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "area":
            return
        attributes = dict(attrs)
        match = SCORE_TITLE_PATTERN.search(attributes.get("title") or "")
        href = attributes.get("href") or ""
        if not match or not href:
            return
        study = parse_qs(urlsplit(href).query).get("study", [])
        if len(study) != 1:
            return
        self.records.append(
            {
                "cell_line": match.group(1).strip(),
                "screener": match.group(2).strip(),
                "study": study[0],
                "official_score": int(match.group(3)),
            }
        )


def _nsc_pair(drug_a: str, drug_b: str) -> tuple[str, str]:
    return tuple(sorted((drug_a, drug_b), key=int))


def _load_nsc_pair_candidates(
    archive_path: Path,
    nsc_to_name: dict[str, str],
) -> dict[tuple[str, str], set[tuple[str, str]]]:
    candidates: dict[tuple[str, str], set[tuple[str, str]]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise ValueError(
                f"Expected one CSV file in the ALMANAC archive; found {len(csv_members)}"
            )
        with archive.open(csv_members[0]) as source:
            chunks = pd.read_csv(
                source,
                usecols=lambda column: column in {"NSC1", "NSC2", "VALID"},
                dtype=str,
                keep_default_na=False,
                chunksize=100_000,
            )
            for chunk in chunks:
                valid = chunk["VALID"].str.strip().str.upper().eq("Y")
                identified = chunk["NSC1"].ne("") & chunk["NSC2"].ne("")
                combinations = chunk.loc[valid & identified, ["NSC1", "NSC2"]].drop_duplicates()
                for nsc1, nsc2 in combinations.itertuples(index=False, name=None):
                    drug_a = nsc_to_name.get(nsc1)
                    drug_b = nsc_to_name.get(nsc2)
                    if drug_a is None or drug_b is None or drug_a == drug_b:
                        continue
                    pair = _canonical_drug_pair(drug_a, drug_b)
                    candidates.setdefault(pair, set()).add(_nsc_pair(nsc1, nsc2))
    return candidates


def _select_stratified_pairs(
    scores: pd.DataFrame,
    nsc_pair_candidates: dict[tuple[str, str], set[tuple[str, str]]],
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    complete = scores.loc[scores["score_complete"]].copy()
    complete["canonical_pair"] = [
        _canonical_drug_pair(drug_a, drug_b)
        for drug_a, drug_b in complete[["drug_a", "drug_b"]].itertuples(
            index=False, name=None
        )
    ]
    group_counts = (
        scores.groupby(["drug_a", "drug_b"])["score_complete"]
        .agg(["size", "sum"])
        .reset_index()
    )
    group_counts["canonical_pair"] = [
        _canonical_drug_pair(drug_a, drug_b)
        for drug_a, drug_b in group_counts[["drug_a", "drug_b"]].itertuples(
            index=False, name=None
        )
    ]
    complete_pairs = set(
        group_counts.loc[group_counts["size"].eq(group_counts["sum"]), "canonical_pair"]
    )
    pair_means = (
        complete.groupby("canonical_pair", as_index=False)
        .agg(mean_combo_score=("combo_score", "mean"))
    )
    pair_means = pair_means.loc[pair_means["canonical_pair"].isin(complete_pairs)].copy()
    pair_means["nsc_pairs"] = pair_means["canonical_pair"].map(nsc_pair_candidates)
    pair_means = pair_means.loc[pair_means["nsc_pairs"].map(len).eq(1)].copy()
    if len(pair_means) < sample_size:
        raise ValueError(
            f"Only {len(pair_means)} complete, uniquely mapped drug pairs are available; "
            f"cannot select {sample_size}"
        )

    pair_means["_pair_sort_key"] = pair_means["canonical_pair"].map(
        lambda pair: "|".join(value.casefold() for value in pair)
    )
    pair_means = pair_means.sort_values(
        ["mean_combo_score", "_pair_sort_key"]
    ).reset_index(drop=True)
    pair_means["score_stratum"] = (
        pd.qcut(
            pair_means["mean_combo_score"].rank(method="first"),
            q=5,
            labels=False,
        )
        .astype(int)
    )
    base_size, remainder = divmod(sample_size, 5)
    selected = []
    for stratum in range(5):
        stratum_size = base_size + int(stratum < remainder)
        candidates = pair_means.loc[pair_means["score_stratum"].eq(stratum)]
        selected.append(
            candidates.sample(
                n=stratum_size,
                random_state=seed + stratum,
            )
        )
    sample = pd.concat(selected, ignore_index=True).sort_values(
        ["score_stratum", "canonical_pair"]
    )
    sample[["nsc1", "nsc2"]] = pd.DataFrame(
        sample["nsc_pairs"].map(lambda pairs: next(iter(pairs))).tolist(),
        index=sample.index,
    )
    sample[["drug_a", "drug_b"]] = pd.DataFrame(
        sample["canonical_pair"].tolist(),
        index=sample.index,
    )
    return sample[
        ["drug_a", "drug_b", "nsc1", "nsc2", "mean_combo_score", "score_stratum"]
    ].reset_index(drop=True)


def _fetch_official_chart(nsc1: str, nsc2: str) -> list[dict[str, Any]]:
    url = f"{OFFICIAL_CHART_ENDPOINT}?{urlencode({'nsc1': nsc1, 'nsc2': nsc2})}"
    response = requests.get(
        url,
        headers={"User-Agent": "pediatric-drug-repurposing ALMANAC score validation"},
        timeout=(10, 45),
    )
    response.raise_for_status()
    parser = _ScoreChartParser()
    parser.feed(response.text)
    if not parser.records:
        raise ValueError(f"No official cell-line scores were found at {response.url}")
    return parser.records


def _compare_pair(
    sample_row: dict[str, Any],
    local_scores: pd.DataFrame,
    official_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    drug_a = sample_row["drug_a"]
    drug_b = sample_row["drug_b"]
    local_frame = local_scores.loc[
        local_scores["drug_a"].eq(drug_a) & local_scores["drug_b"].eq(drug_b)
    ].set_index("cell_line")
    local = local_frame["combo_score"]
    reversal_excluded = (
        local_frame["reversal_qc_excluded"]
        if "reversal_qc_excluded" in local_frame
        else pd.Series(False, index=local_frame.index)
    )
    official = (
        pd.DataFrame(official_records)
        .groupby("cell_line")["official_score"]
        .sum()
    )
    cell_lines = sorted(set(local.index) | set(official.index), key=str.casefold)
    details = []
    for cell_line in cell_lines:
        local_score = local.get(cell_line)
        official_score = official.get(cell_line)
        if local_score is None:
            status = "missing_local"
        elif pd.isna(local_score):
            status = "incomplete_local_score"
        elif official_score is None:
            status = "missing_official"
        elif int(local_score) != int(official_score):
            status = "score_mismatch"
        elif bool(reversal_excluded.get(cell_line, False)):
            status = "exact_match_reversal_excluded"
        else:
            status = "exact_match"
        details.append(
            {
                "drug_a": drug_a,
                "drug_b": drug_b,
                "nsc1": sample_row["nsc1"],
                "nsc2": sample_row["nsc2"],
                "cell_line": cell_line,
                "local_score": local_score,
                "official_score": official_score,
                "reversal_qc_excluded": bool(
                    reversal_excluded.get(cell_line, False)
                ),
                "status": status,
            }
        )
    counts = pd.Series([detail["status"] for detail in details]).value_counts()
    summary = {
        "drug_a": drug_a,
        "drug_b": drug_b,
        "nsc1": sample_row["nsc1"],
        "nsc2": sample_row["nsc2"],
        "mean_combo_score_stratum": int(sample_row["score_stratum"]),
        "local_cell_line_groups": int(local.size),
        "official_cell_line_groups": int(official.size),
        "compared_cell_line_groups": len(details),
        "exact_matches": int(counts.get("exact_match", 0)),
        "exact_matches_reversal_excluded": int(
            counts.get("exact_match_reversal_excluded", 0)
        ),
        "score_mismatches": int(counts.get("score_mismatch", 0)),
        "incomplete_local_scores": int(counts.get("incomplete_local_score", 0)),
        "missing_local": int(counts.get("missing_local", 0)),
        "missing_official": int(counts.get("missing_official", 0)),
        "source_url": f"{OFFICIAL_CHART_ENDPOINT}?{urlencode({'nsc1': sample_row['nsc1'], 'nsc2': sample_row['nsc2']})}",
    }
    return summary, details


def validate_nci_almanac_official_scores(
    archive_path: str | Path | None = None,
    compound_names_path: str | Path | None = None,
    score_path: str | Path | None = None,
    reversal_exclusions_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    sample_size: int = SAMPLE_SIZE,
    seed: int = SAMPLE_SEED,
    workers: int = 3,
) -> dict[str, Path]:
    raw_source = Path(archive_path or Path(CONFIG["raw_data_path"]) / "almanac" / ALMANAC_ARCHIVE_FILENAME)
    names_source = Path(
        compound_names_path
        or Path(CONFIG["raw_data_path"]) / "almanac" / ALMANAC_COMPOUND_NAMES_FILENAME
    )
    score_source = Path(
        score_path
        or Path(CONFIG["processed_data_path"]) / "almanac" / "nci_almanac_combo_scores.csv"
    )
    reversal_exclusions_source = Path(
        reversal_exclusions_path
        or Path(CONFIG["processed_data_path"])
        / "almanac"
        / "nci_almanac_reversal_exclusions.csv"
    )
    manifest_source = Path(
        manifest_path
        or Path(CONFIG["processed_data_path"]) / "almanac" / "nci_almanac_manifest.json"
    )
    destination = Path(output_dir or score_source.parent)
    destination.mkdir(parents=True, exist_ok=True)

    nsc_to_name, _ = _load_compound_map(names_source)
    nsc_pair_candidates = _load_nsc_pair_candidates(raw_source, nsc_to_name)
    local_scores = pd.read_csv(score_source)
    sample = _select_stratified_pairs(
        local_scores,
        nsc_pair_candidates,
        sample_size=sample_size,
        seed=seed,
    )
    if not reversal_exclusions_source.is_file():
        raise FileNotFoundError(
            "ALMANAC reversal-exclusion scores are required for an unbiased "
            f"official-chart comparison: {reversal_exclusions_source}"
        )
    reversal_exclusions = pd.read_csv(reversal_exclusions_source)
    local_scores["reversal_qc_excluded"] = False
    reversal_exclusions["reversal_qc_excluded"] = True
    comparison_scores = pd.concat(
        [local_scores, reversal_exclusions],
        ignore_index=True,
    )
    fetched: dict[int, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(
                _fetch_official_chart,
                row.nsc1,
                row.nsc2,
            ): index
            for index, row in enumerate(sample.itertuples(index=False))
        }
        try:
            for future in as_completed(future_to_index):
                fetched[future_to_index[future]] = future.result()
        except Exception:
            for future in future_to_index:
                future.cancel()
            raise

    pair_summaries = []
    comparison_details = []
    for index, sample_row in enumerate(sample.to_dict("records")):
        summary, details = _compare_pair(
            sample_row,
            comparison_scores,
            fetched[index],
        )
        pair_summaries.append(summary)
        comparison_details.extend(details)

    pair_results = pd.DataFrame(pair_summaries)
    detail_results = pd.DataFrame(comparison_details)
    summary_path = destination / "nci_almanac_official_validation.json"
    pair_path = destination / "nci_almanac_official_validation_pairs.csv"
    detail_path = destination / "nci_almanac_official_validation_details.csv"
    pair_results.to_csv(pair_path, index=False)
    detail_results.to_csv(detail_path, index=False)
    exact_matches = int(pair_results["exact_matches"].sum())
    exact_reversal_excluded = int(
        pair_results["exact_matches_reversal_excluded"].sum()
    )
    score_mismatches = int(pair_results["score_mismatches"].sum())
    incomplete_local_scores = int(pair_results["incomplete_local_scores"].sum())
    missing_local = int(pair_results["missing_local"].sum())
    missing_official = int(pair_results["missing_official"].sum())
    score_validation_passed = (
        len(pair_results) == sample_size
        and score_mismatches == 0
        and incomplete_local_scores == 0
        and missing_local == 0
    )
    official_chart_coverage_complete = missing_official == 0
    summary: dict[str, Any] = {
        "source": "NCI DTP official per-pair detailed score charts",
        "source_url": OFFICIAL_CHART_ENDPOINT,
        "sample_size": sample_size,
        "seed": seed,
        "stratification": (
            "Five equally sized strata of mean cell-line ComboScore; pairs must have "
            "complete local scores and a unique NSC pair identity."
        ),
        "stratum_pair_counts": {
            str(index): int(count)
            for index, count in pair_results["mean_combo_score_stratum"]
            .value_counts()
            .sort_index()
            .items()
        },
        "official_charts_retrieved": len(pair_results),
        "cell_line_pair_groups_compared": int(pair_results["compared_cell_line_groups"].sum()),
        "exact_cell_line_pair_score_matches": exact_matches + exact_reversal_excluded,
        "retained_score_matches": exact_matches,
        "exact_matches_in_reversal_exclusions": exact_reversal_excluded,
        "score_mismatches": score_mismatches,
        "incomplete_local_scores": incomplete_local_scores,
        "missing_local_groups": missing_local,
        "missing_official_groups": missing_official,
        "validation_passed": score_validation_passed,
        "official_chart_coverage_complete": official_chart_coverage_complete,
        "full_dataset_validated": False,
        "training_ready": False,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "pairs": str(pair_path),
            "details": str(detail_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    if manifest_source.is_file():
        manifest = json.loads(manifest_source.read_text())
        manifest["official_score_validation"] = summary
        manifest["independent_combo_score_validation"] = score_validation_passed
        manifest["training_ready"] = False
        temporary_manifest = manifest_source.with_suffix(".json.tmp")
        try:
            temporary_manifest.write_text(json.dumps(manifest, indent=2))
            temporary_manifest.replace(manifest_source)
        finally:
            if temporary_manifest.exists():
                temporary_manifest.unlink()

    return {
        "summary": summary_path,
        "pairs": pair_path,
        "details": detail_path,
    }


if __name__ == "__main__":
    outputs = validate_nci_almanac_official_scores()
    for name, path in outputs.items():
        print(f"{name}: {path}")
