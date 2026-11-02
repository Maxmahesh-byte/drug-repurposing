from pathlib import Path

import pandas as pd

from src.ingestion.gdsc2 import build_gdsc2_response_dataset


def build_phase3_dataset(
    output_dir: str | Path | None = None,
    raw_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compatibility wrapper returning real GDSC2 cell-line response data."""
    output_path = build_gdsc2_response_dataset(output_dir=output_dir, raw_path=raw_path)
    return pd.read_csv(output_path)
