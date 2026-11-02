from pathlib import Path
import pandas as pd

TARGET_COLUMNS = [
    "sample_id", "tumor_type", "gene_symbol",
    "molecular_layer", "measurement", "score", "source",
]

def build_phase2_dataset(output_dir=None, phase1_df=None) -> pd.DataFrame:
    """Block generated sample IDs and unverified TARGET annotations."""
    raise RuntimeError(
        "TARGET ingestion is blocked: hard-coded examples and generated "
        "TARGET sample IDs are not verified source records. Use authoritative "
        "records and validated sample identifiers before ingestion."
    )

if __name__ == "__main__":
    build_phase2_dataset()
