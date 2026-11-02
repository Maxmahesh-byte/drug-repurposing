from pathlib import Path

import pandas as pd


PHASE4_COLUMNS = [
    "drug_name",
    "target_gene",
    "pathway",
    "interaction_type",
    "evidence_score",
    "source",
]


def build_phase4_dataset(output_dir: str | Path | None = None) -> pd.DataFrame:
    """Refuse to generate example rows as if they were sourced evidence."""
    raise RuntimeError(
        "Phase 4 ingestion is blocked: the existing implementation contains "
        "hard-coded examples, not verified ChEMBL, Reactome, or STRING records. "
        "A source-backed importer with original record identifiers and provenance "
        "must be implemented before this dataset can be generated."
    )


if __name__ == "__main__":
    build_phase4_dataset()
