#!/usr/bin/env python3
"""Build ChEMBL-derived chemical features aligned to the compact GDSC drug graph.

The output is response-label-free: only the audited GDSC-to-ChEMBL mapping and
ChEMBL molecule structure records are used. No response values enter this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = ROOT / "data/raw/chembl/gdsc2_drug_molecule_mappings.csv"
DEFAULT_GRAPH = ROOT / "data/processed/graph_v2/typed_graph_v2.pt"
DEFAULT_OUTPUT = ROOT / "data/ai/chemical_features"
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
MORGAN_BITS = 2048
MORGAN_RADIUS = 2
DESCRIPTOR_NAMES = (
    "MolWt", "MolLogP", "TPSA", "HBD", "HBA", "RotatableBonds",
    "RingCount", "FractionCSP3",
)
FEATURE_NAMES = tuple(
    [f"morgan_r{MORGAN_RADIUS}_bit_{i}" for i in range(MORGAN_BITS)]
    + list(DESCRIPTOR_NAMES)
    + ["structure_available"]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "drug-repurposing/0.1 (chemical feature audit)"})
    return session


def fetch_molecule(session: requests.Session, molecule_id: str) -> dict[str, Any]:
    response = session.get(
        f"{CHEMBL_API}/molecule/{molecule_id}.json", timeout=(5, 40)
    )
    response.raise_for_status()
    payload = response.json()
    returned_id = str(payload.get("molecule_chembl_id") or "")
    if returned_id != molecule_id:
        raise ValueError(
            f"ChEMBL returned molecule {returned_id!r} for requested {molecule_id!r}."
        )
    return payload


def structure_features(
    canonical_smiles: str | None,
    generator,
) -> tuple[np.ndarray, bool]:
    """Return Morgan bits + descriptors; missing structures are explicit zeros."""
    width = MORGAN_BITS + len(DESCRIPTOR_NAMES) + 1
    if not canonical_smiles:
        return np.zeros(width, dtype=np.float32), False

    molecule = Chem.MolFromSmiles(str(canonical_smiles))
    if molecule is None:
        return np.zeros(width, dtype=np.float32), False

    fingerprint = generator.GetFingerprint(molecule)
    bits = np.zeros((MORGAN_BITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, bits)
    descriptors = np.asarray(
        [
            Descriptors.MolWt(molecule),
            Descriptors.MolLogP(molecule),
            rdMolDescriptors.CalcTPSA(molecule),
            Lipinski.NumHDonors(molecule),
            Lipinski.NumHAcceptors(molecule),
            Lipinski.NumRotatableBonds(molecule),
            rdMolDescriptors.CalcNumRings(molecule),
            rdMolDescriptors.CalcFractionCSP3(molecule),
        ],
        dtype=np.float32,
    )
    if not np.isfinite(descriptors).all():
        raise ValueError("RDKit produced non-finite molecular descriptors.")
    return np.concatenate(
        [bits, descriptors, np.ones((1,), dtype=np.float32)]
    ), True


def load_graph_drug_ids(graph_path: Path) -> list[str]:
    # Graph artifacts are trusted project outputs; PyG stores HeteroData objects.
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    if "drug" not in set(graph.node_types):
        raise ValueError("Typed graph has no 'drug' node type.")
    ids = [str(value).strip() for value in graph["drug"].node_ids]
    if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Graph drug node IDs must be non-empty and unique.")
    return ids


def validate_mapping(mapping: pd.DataFrame) -> pd.DataFrame:
    required = {"gdsc_drug_id", "molecule_chembl_id", "provenance_state"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Mapping table missing columns: {sorted(missing)}")
    result = mapping.copy()
    for column in ("gdsc_drug_id", "molecule_chembl_id"):
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq("").any():
            raise ValueError(f"Mapping column {column} contains blank IDs.")
    if result["gdsc_drug_id"].duplicated().any():
        raise ValueError("A GDSC drug maps to multiple rows; resolve ambiguity first.")
    if not result["provenance_state"].astype(str).str.contains(
        "exact_string_match_not_chemical_equivalence", regex=False
    ).all():
        raise ValueError("Unexpected mapping provenance; refusing un-audited mappings.")
    return result


def build_features(
    mapping_path: Path,
    graph_path: Path,
    output_dir: Path,
    pause_seconds: float = 0.10,
) -> dict[str, Any]:
    for path, label in ((mapping_path, "mapping table"), (graph_path, "typed graph")):
        if not path.is_file():
            raise FileNotFoundError(f"Required {label} not found: {path}")
    if pause_seconds < 0:
        raise ValueError("pause_seconds cannot be negative.")

    mapping = validate_mapping(pd.read_csv(mapping_path, dtype=str))
    graph_drug_ids = load_graph_drug_ids(graph_path)
    mapping_by_id = mapping.set_index("gdsc_drug_id", drop=False)
    graph_id_set = set(graph_drug_ids)
    mapping_id_set = set(mapping_by_id.index)
    missing_mappings = sorted(graph_id_set - mapping_id_set)
    extra_mappings = sorted(mapping_id_set - graph_id_set)
    if missing_mappings:
        raise ValueError(
            "Some graph drug IDs have no audited ChEMBL mapping. "
            f"First IDs: {missing_mappings[:10]}. Do not infer chemical identity."
        )

    RDLogger.DisableLog("rdApp.*")
    generator = GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    session = make_session()
    molecule_cache: dict[str, dict[str, Any]] = {}
    feature_rows: list[np.ndarray] = []
    audit_rows: list[dict[str, Any]] = []

    for position, gdsc_id in enumerate(graph_drug_ids, start=1):
        row = mapping_by_id.loc[gdsc_id]
        chembl_id = str(row["molecule_chembl_id"])
        if chembl_id not in molecule_cache:
            molecule_cache[chembl_id] = fetch_molecule(session, chembl_id)
            if pause_seconds:
                time.sleep(pause_seconds)
        molecule = molecule_cache[chembl_id]
        structures = molecule.get("molecule_structures") or {}
        smiles = structures.get("canonical_smiles")
        vector, available = structure_features(smiles, generator)
        feature_rows.append(vector)
        audit_rows.append(
            {
                "graph_drug_index": position - 1,
                "gdsc_drug_id": gdsc_id,
                "gdsc_drug_name": str(row.get("gdsc_drug_name", "")),
                "molecule_chembl_id": chembl_id,
                "drug_match_method": str(row.get("drug_match_method", "")),
                "mapping_provenance": str(row["provenance_state"]),
                "canonical_smiles": smiles or "",
                "standard_inchi": structures.get("standard_inchi") or "",
                "standard_inchi_key": structures.get("standard_inchi_key") or "",
                "structure_available": bool(available),
                "structure_source_url": f"{CHEMBL_API}/molecule/{chembl_id}.json",
            }
        )
        if position % 25 == 0:
            print(f"Processed {position}/{len(graph_drug_ids)} graph drugs", flush=True)

    matrix = np.stack(feature_rows).astype(np.float32)
    if matrix.shape != (len(graph_drug_ids), len(FEATURE_NAMES)):
        raise RuntimeError("Chemical feature matrix has an unexpected shape.")
    if not np.isfinite(matrix).all():
        raise ValueError("Chemical feature matrix contains non-finite values.")

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "gdsc2_chemical_features.npz"
    audit_path = output_dir / "gdsc2_chemical_feature_audit.csv"
    manifest_path = output_dir / "gdsc2_chemical_feature_manifest.json"
    np.savez_compressed(
        npz_path,
        drug_ids=np.asarray(graph_drug_ids, dtype=str),
        features=matrix,
        feature_names=np.asarray(FEATURE_NAMES, dtype=str),
    )
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    manifest = {
        "source": "ChEMBL molecule API",
        "structure_fields": [
            "canonical_smiles", "standard_inchi", "standard_inchi_key"
        ],
        "fingerprint": {
            "type": "Morgan",
            "radius": MORGAN_RADIUS,
            "bits": MORGAN_BITS,
        },
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "feature_width": len(FEATURE_NAMES),
        "graph_drug_count": len(graph_drug_ids),
        "mapped_graph_drug_count": len(graph_drug_ids),
        "mapping_rows_not_in_graph": extra_mappings,
        "structure_available_count": int(
            pd.DataFrame(audit_rows)["structure_available"].sum()
        ),
        "structure_missing_count": int(
            (~pd.DataFrame(audit_rows)["structure_available"]).sum()
        ),
        "alignment": "NPZ row i corresponds exactly to drug_ids[i] and graph drug node i",
        "response_labels_used": False,
        "mapping_sha256": sha256_file(mapping_path),
        "typed_graph_sha256": sha256_file(graph_path),
        "outputs": {
            "features": str(npz_path),
            "audit": str(audit_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pause-seconds", type=float, default=0.10)
    args = parser.parse_args()
    print(json.dumps(
        build_features(args.mapping, args.graph, args.output_dir, args.pause_seconds),
        indent=2,
    ))


if __name__ == "__main__":
    main()
