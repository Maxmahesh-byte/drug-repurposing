from pathlib import Path
import pandas as pd

PHASE6_COLUMNS = [
    "drug_name", "target_gene", "pdb_id",
    "binding_mode", "binding_score", "source",
]

def build_phase6_dataset(output_dir=None) -> pd.DataFrame:
    """Block generation of unverified example structural claims."""
    raise RuntimeError(
        "Phase 6 ingestion is blocked: hard-coded examples are not verified "
        "PDB ligand/structure/target records. Verify each structure and "
        "record its source identifiers before ingestion."
    )

if __name__ == "__main__":
    build_phase6_dataset()
