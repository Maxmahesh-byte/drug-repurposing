from pathlib import Path
import pandas as pd

PHASE7_COLUMNS = [
    "drug_name", "tumor_type", "assay_type",
    "validation_result", "confidence_score", "source",
]

def build_phase7_dataset(output_dir=None) -> pd.DataFrame:
    """Block generation of unverified experimental-validation claims."""
    raise RuntimeError(
        "Phase 7 ingestion is blocked: hard-coded examples are not verified "
        "experimental records. Require traceable study/assay identifiers "
        "and source provenance before ingestion."
    )

if __name__ == "__main__":
    build_phase7_dataset()
