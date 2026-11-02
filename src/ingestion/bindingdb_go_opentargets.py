from pathlib import Path
import pandas as pd

PHASE5_COLUMNS = [
    "drug_name", "target_gene", "gene_ontology_term",
    "evidence_type", "confidence_score", "source",
]

def build_phase5_dataset(output_dir=None) -> pd.DataFrame:
    """Block generation of unverified example target-evidence rows."""
    raise RuntimeError(
        "Phase 5 ingestion is blocked: hard-coded examples are not verified "
        "BindingDB, Gene Ontology, or Open Targets records. Implement "
        "source-backed ingestion with original record identifiers and provenance."
    )

if __name__ == "__main__":
    build_phase5_dataset()
