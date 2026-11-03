from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.config import CONFIG

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
OUTPUT_COLUMNS = [
    "gdsc_drug_id",
    "gdsc_drug_name",
    "molecule_chembl_id",
    "drug_match_method",
    "matched_chembl_synonym",
    "chembl_synonym_type",
    "target_chembl_id",
    "target_name",
    "target_type",
    "uniprot_accession",
    "gene_symbol",
    "mechanism_of_action",
    "action_type",
    "direct_interaction",
    "max_phase",
    "source_url",
]

MAPPING_OUTPUT_COLUMNS = [
    "gdsc_drug_id",
    "gdsc_drug_name",
    "molecule_chembl_id",
    "drug_match_method",
    "matched_chembl_synonym",
    "chembl_synonym_type",
    "mapping_source_url",
    "provenance_state",
]


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "drug-repurposing/0.1 (ChEMBL evidence ingestion)"})
    return session


def _get_json(session: requests.Session, resource: str, params: dict[str, object] | None = None) -> dict:
    response = session.get(f"{CHEMBL_API}/{resource}.json", params=params, timeout=(5, 30))
    response.raise_for_status()
    return response.json()


def _get_collection(
    session: requests.Session,
    resource: str,
    collection_key: str,
    params: dict[str, object],
) -> list[dict]:
    results = []
    offset = 0
    limit = 1000
    while True:
        page = _get_json(session, resource, {**params, "limit": limit, "offset": offset})
        rows = page.get(collection_key, [])
        results.extend(rows)
        total = (page.get("page_meta") or {}).get("total_count")
        if not rows or (total is not None and len(results) >= total) or len(rows) < limit:
            return results
        offset += len(rows)


def _chunks(values: list[str], size: int = 50):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def select_exact_molecules(drug_name: str, molecules: list[dict]) -> list[dict]:
    normalized_name = drug_name.strip().casefold()
    return [
        molecule
        for molecule in molecules
        if (molecule.get("pref_name") or "").strip().casefold() == normalized_name
    ]


def select_exact_synonym_molecules(drug_name: str, molecules: list[dict]) -> list[tuple[dict, str, str]]:
    normalized_name = drug_name.strip().casefold()
    matches = {}
    for molecule in molecules:
        molecule_id = molecule.get("molecule_chembl_id")
        if not molecule_id:
            continue
        for synonym in molecule.get("molecule_synonyms") or []:
            synonym_name = (synonym.get("molecule_synonym") or "").strip()
            if synonym_name.casefold() == normalized_name:
                match = (molecule, synonym_name, synonym.get("syn_type") or "")
                previous = matches.get(molecule_id)
                if previous is None or (match[2], match[1]) < (previous[2], previous[1]):
                    matches[molecule_id] = match
    return [matches[molecule_id] for molecule_id in sorted(matches)]


def extract_gene_symbols(target: dict) -> list[tuple[str, str]]:
    genes = []
    for component in target.get("target_components") or []:
        accession = component.get("accession") or ""
        for synonym in component.get("target_component_synonyms") or []:
            if synonym.get("syn_type") == "GENE_SYMBOL" and synonym.get("component_synonym"):
                genes.append((accession, synonym["component_synonym"]))
    return sorted(set(genes))


def build_molecule_mapping_table(
    drug_pairs_by_molecule: dict[str, list[tuple[str, str, str, str, str]]],
) -> pd.DataFrame:
    """Return every uniquely resolved GDSC2-drug to ChEMBL-molecule mapping.

    This table is intentionally independent of whether ChEMBL has a direct
    human-target mechanism for the molecule.
    """
    rows = []
    for molecule_id, drug_pairs in sorted(drug_pairs_by_molecule.items()):
        for drug_id, drug_name, match_method, matched_synonym, synonym_type in drug_pairs:
            rows.append(
                {
                    "gdsc_drug_id": str(drug_id),
                    "gdsc_drug_name": str(drug_name),
                    "molecule_chembl_id": str(molecule_id),
                    "drug_match_method": str(match_method),
                    "matched_chembl_synonym": str(matched_synonym),
                    "chembl_synonym_type": str(synonym_type),
                    "mapping_source_url": f"{CHEMBL_API}/molecule/{molecule_id}.json",
                    "provenance_state": "exact_string_match_not_chemical_equivalence",
                }
            )
    result = pd.DataFrame(rows, columns=MAPPING_OUTPUT_COLUMNS)
    if not result.empty and result["gdsc_drug_id"].duplicated().any():
        raise ValueError("A GDSC2 drug resolved to more than one ChEMBL molecule.")
    return result.drop_duplicates().reset_index(drop=True)


def build_chembl_target_dataset(
    gdsc_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    response_path = Path(gdsc_path) if gdsc_path is not None else Path(CONFIG["raw_data_path"]) / "gdsc" / "gdsc2_responses.csv"
    if not response_path.is_file():
        raise FileNotFoundError(f"Real GDSC2 response file is required: {response_path}")

    dataset_dir = Path(output_dir) if output_dir is not None else Path(CONFIG["raw_data_path"]) / "chembl"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    drug_pairs = pd.read_csv(response_path, usecols=["drug_id", "drug_name"], dtype=str).drop_duplicates()
    session = _session()
    target_cache: dict[str, dict] = {}
    output_rows = []
    unmatched_drugs = []
    ambiguous_drugs = []
    nonhuman_targets = set()
    names = sorted(drug_pairs["drug_name"].dropna().astype(str).unique().tolist())
    matches_by_drug: dict[str, list[tuple[dict, str, str, str]]] = {}

    for batch_number, name_batch in enumerate(list(_chunks(names)), start=1):
        molecule_rows = _get_collection(
            session,
            "molecule",
            "molecules",
            {"pref_name__in": ",".join(name.upper() for name in name_batch)},
        )
        for drug_name in name_batch:
            preferred_matches = select_exact_molecules(drug_name, molecule_rows)
            if preferred_matches:
                matches_by_drug[drug_name] = [
                    (molecule, "exact_preferred_name", "", "")
                    for molecule in preferred_matches
                ]
        print(f"ChEMBL molecule batches: {batch_number}/{(len(names) + 49) // 50}", flush=True)

    synonym_names = [name for name in names if name not in matches_by_drug]
    for batch_number, name_batch in enumerate(list(_chunks(synonym_names)), start=1):
        molecule_rows = _get_collection(
            session,
            "molecule",
            "molecules",
            {"molecule_synonyms__molecule_synonym__in": ",".join(name_batch)},
        )
        for drug_name in name_batch:
            synonym_matches = select_exact_synonym_molecules(drug_name, molecule_rows)
            if synonym_matches:
                matches_by_drug[drug_name] = [
                    (molecule, "exact_synonym", synonym_name, synonym_type)
                    for molecule, synonym_name, synonym_type in synonym_matches
                ]
        print(f"ChEMBL synonym batches: {batch_number}/{(len(synonym_names) + 49) // 50}", flush=True)

    drug_pairs_by_molecule: dict[str, list[tuple[str, str, str, str, str]]] = {}
    preferred_match_pairs = set()
    synonym_match_pairs = set()
    for drug_id, drug_name in drug_pairs.itertuples(index=False, name=None):
        matches = matches_by_drug.get(drug_name, [])
        if not matches:
            unmatched_drugs.append({"drug_id": drug_id, "drug_name": drug_name})
            continue
        molecule_ids = {molecule.get("molecule_chembl_id") for molecule, _, _, _ in matches}
        molecule_ids.discard(None)
        if len(molecule_ids) != 1:
            ambiguous_drugs.append({"drug_id": drug_id, "drug_name": drug_name, "molecule_chembl_ids": sorted(molecule_ids)})
            continue
        for molecule, match_method, matched_synonym, synonym_type in matches:
            molecule_id = molecule.get("molecule_chembl_id")
            if molecule_id:
                drug_pairs_by_molecule.setdefault(molecule_id, []).append(
                    (drug_id, drug_name, match_method, matched_synonym, synonym_type)
                )
                if match_method == "exact_preferred_name":
                    preferred_match_pairs.add((drug_id, drug_name))
                else:
                    synonym_match_pairs.add((drug_id, drug_name))

    molecule_ids = sorted(drug_pairs_by_molecule)
    mapping_result = build_molecule_mapping_table(drug_pairs_by_molecule)
    if mapping_result.empty:
        raise ValueError("No unambiguous exact-name GDSC2-to-ChEMBL molecule mappings were resolved.")
    mechanisms = []
    for batch_number, molecule_batch in enumerate(list(_chunks(molecule_ids)), start=1):
        mechanisms.extend(
            _get_collection(
                session,
                "mechanism",
                "mechanisms",
                {"molecule_chembl_id__in": ",".join(molecule_batch)},
            )
        )
        print(f"ChEMBL mechanism batches: {batch_number}/{(len(molecule_ids) + 49) // 50}", flush=True)

    direct_mechanisms = [
        mechanism
        for mechanism in mechanisms
        if int(mechanism.get("direct_interaction") or 0) == 1
        and mechanism.get("target_chembl_id")
        and mechanism.get("molecule_chembl_id") in drug_pairs_by_molecule
    ]
    target_ids = sorted({mechanism["target_chembl_id"] for mechanism in direct_mechanisms})
    for batch_number, target_batch in enumerate(list(_chunks(target_ids)), start=1):
        for target in _get_collection(
            session,
            "target",
            "targets",
            {"target_chembl_id__in": ",".join(target_batch)},
        ):
            target_cache[target["target_chembl_id"]] = target
        print(f"ChEMBL target batches: {batch_number}/{(len(target_ids) + 49) // 50}", flush=True)

    for mechanism in direct_mechanisms:
        molecule_id = mechanism["molecule_chembl_id"]
        target_id = mechanism["target_chembl_id"]
        target = target_cache.get(target_id, {})
        if target.get("organism") != "Homo sapiens":
            nonhuman_targets.add(target_id)
            continue
        gene_components = extract_gene_symbols(target)
        if not gene_components:
            gene_components = [("", "")]

        for drug_id, drug_name, match_method, matched_synonym, synonym_type in drug_pairs_by_molecule[molecule_id]:
            for accession, gene_symbol in gene_components:
                output_rows.append(
                    {
                        "gdsc_drug_id": drug_id,
                        "gdsc_drug_name": drug_name,
                        "molecule_chembl_id": molecule_id,
                        "drug_match_method": match_method,
                        "matched_chembl_synonym": matched_synonym,
                        "chembl_synonym_type": synonym_type,
                        "target_chembl_id": target_id,
                        "target_name": target.get("pref_name"),
                        "target_type": target.get("target_type"),
                        "uniprot_accession": accession,
                        "gene_symbol": gene_symbol,
                        "mechanism_of_action": mechanism.get("mechanism_of_action"),
                        "action_type": mechanism.get("action_type"),
                        "direct_interaction": True,
                        "max_phase": mechanism.get("max_phase"),
                        "source_url": f"{CHEMBL_API}/mechanism/{mechanism.get('mec_id')}.json",
                    }
                )

    # An empty target-evidence table is valid when molecule mappings exist
    # but none of the matched compounds has a qualifying direct human mechanism.
    result = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS).drop_duplicates()
    mechanism_pairs = set(zip(result["gdsc_drug_id"], result["gdsc_drug_name"]))
    mapped_pairs = preferred_match_pairs | synonym_match_pairs
    exact_matches_without_mechanisms = sorted(mapped_pairs - mechanism_pairs)
    mapping_output_path = dataset_dir / "gdsc2_drug_molecule_mappings.csv"
    mapping_temporary_path = mapping_output_path.with_suffix(".csv.tmp")
    try:
        mapping_result.to_csv(mapping_temporary_path, index=False)
        mapping_temporary_path.replace(mapping_output_path)
    finally:
        if mapping_temporary_path.exists():
            mapping_temporary_path.unlink()

    output_path = dataset_dir / "gdsc2_chembl_target_evidence.csv"
    temporary_path = output_path.with_suffix(".csv.tmp")
    try:
        result.to_csv(temporary_path, index=False)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    manifest = {
        "source": "ChEMBL REST API",
        "data_origin": "official_chembl_mechanism_records",
        "api_base": CHEMBL_API,
        "matching_policy": "exact preferred-name or exact ChEMBL synonym; ambiguous matches excluded; direct human-target mechanisms only",
        "gdsc_unique_drug_pairs": int(len(drug_pairs)),
        "exact_preferred_name_matches": len(preferred_match_pairs),
        "exact_synonym_matches": len(synonym_match_pairs),
        "exactly_matched_drug_pairs": len(mapped_pairs),
        "drug_pairs_with_direct_human_mechanisms": len(mechanism_pairs),
        "unmatched_drugs": unmatched_drugs,
        "ambiguous_drugs": ambiguous_drugs,
        "exact_matches_without_direct_human_mechanisms": [
            {"drug_id": drug_id, "drug_name": drug_name}
            for drug_id, drug_name in exact_matches_without_mechanisms
        ],
        "nonhuman_target_ids_excluded": sorted(nonhuman_targets),
        "molecule_mapping_rows": int(len(mapping_result)),
        "unique_mapped_gdsc_drugs": int(mapping_result["gdsc_drug_id"].nunique()),
        "molecule_mapping_output_path": str(mapping_output_path),
        "mechanism_gene_rows": int(len(result)),
        "unique_chembl_molecules": int(result["molecule_chembl_id"].nunique()),
        "unique_human_targets": int(result["target_chembl_id"].nunique()),
        "unique_gene_symbols": int(result["gene_symbol"].replace("", pd.NA).nunique()),
        "columns": list(result.columns),
        "output_path": str(output_path),
    }
    (dataset_dir / "gdsc2_chembl_manifest.json").write_text(json.dumps(manifest, indent=2))
    return output_path