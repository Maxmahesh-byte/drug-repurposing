from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

SAMPLES_PATH = PROCESSED_DIR / "pediatric_samples.tsv"
GRAPH_PATH = PROCESSED_DIR / "pediatric_kg.pt"
MAPPINGS_PATH = METADATA_DIR / "pediatric_kg_mappings.json"


def _encode(values: pd.Series) -> tuple[torch.Tensor, dict[str, int]]:
    unique = sorted(values.dropna().astype(str).unique().tolist())
    mapping = {value: idx for idx, value in enumerate(unique)}
    encoded = torch.tensor(
        [mapping[str(value)] for value in values],
        dtype=torch.long,
    )
    return encoded, mapping


def build_pediatric_graph() -> HeteroData:
    if not SAMPLES_PATH.exists():
        raise FileNotFoundError(
            f"{SAMPLES_PATH} not found. Run src/ingestion/data_ingest.py first."
        )

    df = pd.read_csv(SAMPLES_PATH, sep="\t", dtype=str)
    if df.empty:
        raise ValueError("Standardized sample table is empty.")

    sample_codes, sample_map = _encode(df["sample_id"])
    disease_codes, disease_map = _encode(df["disease"])

    data = HeteroData()
    data["sample"].num_nodes = len(sample_map)
    data["disease"].num_nodes = len(disease_map)

    # Sample -> disease is a direct biological-context relation from harmonized metadata.
    edges = torch.stack([sample_codes, disease_codes], dim=0)
    data["sample", "has_diagnosis", "disease"].edge_index = edges
    data["disease", "rev_has_diagnosis", "sample"].edge_index = edges.flip(0)

    # Non-random sample features: simple data-availability indicators where available.
    feature_cols = [c for c in ("has_rnaseq", "has_dna") if c in df.columns]
    if feature_cols:
        ordered = (
            df.drop_duplicates("sample_id")
            .set_index("sample_id")
            .reindex(list(sample_map.keys()))
        )
        x = torch.tensor(
            ordered[feature_cols].fillna("0").astype(float).to_numpy(),
            dtype=torch.float32,
        )
        data["sample"].x = x

    # Store labels/mappings separately so model code never depends on row order by accident.
    mappings = {
        "sample": sample_map,
        "disease": disease_map,
        "feature_columns": feature_cols,
    }

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    MAPPINGS_PATH.write_text(json.dumps(mappings, indent=2) + "\n")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(data, GRAPH_PATH)

    print(f"Saved graph to {GRAPH_PATH}")
    print(f"Samples: {data['sample'].num_nodes:,}")
    print(f"Disease nodes: {data['disease'].num_nodes:,}")
    print(
        "Sample→disease edges:",
        data["sample", "has_diagnosis", "disease"].edge_index.shape[1],
    )

    return data


if __name__ == "__main__":
    build_pediatric_graph()
