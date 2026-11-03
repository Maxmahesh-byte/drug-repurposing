import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.bindingdb_go_opentargets import build_phase5_dataset
from src.ingestion.chembl_reactome_string import build_phase4_dataset
from src.ingestion.experimental_validation import build_phase7_dataset
from src.ingestion.gdsc_pediatric import build_phase3_dataset
from src.ingestion.openpedcan import build_phase1_dataset
from src.ingestion.pdb_structural import build_phase6_dataset
from src.ingestion.openpedcan_molecular import build_openpedcan_mutation_dataset
from src.ingestion.openpedcan_additional import (
    build_openpedcan_copy_number_dataset,
    build_openpedcan_fusion_dataset,
)
from src.ingestion.gdsc2 import build_gdsc2_response_dataset
from src.ingestion.chembl_targets import build_chembl_target_dataset
from src.utils.config import CONFIG


def download_gdsc_data():
    """Ingest the official Wellcome Sanger GDSC2 fitted response workbook."""
    path = build_gdsc2_response_dataset()
    print(f"Official GDSC2 response dataset saved at {path}")
    return path


def ingest_chembl_target_evidence():
    """Ingest exact-name ChEMBL mechanisms for official GDSC2 compounds."""
    path = build_chembl_target_dataset()
    print(f"ChEMBL molecule mappings and direct human-target evidence saved under {path.parent}")
    return path


def ingest_phase1_openpedcan():
    """Create the Phase 1 pediatric cancer evidence table from OpenPedCan."""
    openpedcan_dir = Path(CONFIG["raw_data_path"]) / "openpedcan"
    df = build_phase1_dataset(output_dir=openpedcan_dir)
    print(f"✅ OpenPedCan Phase 1 dataset saved: {len(df)} rows at {openpedcan_dir}")
    return df


def ingest_phase2_openpedcan_mutations():
    """Ingest real OpenPedCan consensus variants joined to clinical biospecimens."""
    openpedcan_dir = Path(CONFIG["raw_data_path"]) / "openpedcan"
    path = build_openpedcan_mutation_dataset(output_dir=openpedcan_dir)
    print(f"OpenPedCan Phase 2 mutation dataset saved at {path}")
    return path


def ingest_phase3_openpedcan_copy_number():
    """Ingest real focal copy-number calls with clinical cohort metadata."""
    openpedcan_dir = Path(CONFIG["raw_data_path"]) / "openpedcan"
    path = build_openpedcan_copy_number_dataset(output_dir=openpedcan_dir)
    print(f"OpenPedCan Phase 3 copy-number dataset saved at {path}")
    return path


def ingest_phase4_openpedcan_fusions():
    """Ingest real filtered fusions with clinical cohort metadata."""
    openpedcan_dir = Path(CONFIG["raw_data_path"]) / "openpedcan"
    path = build_openpedcan_fusion_dataset(output_dir=openpedcan_dir)
    print(f"OpenPedCan Phase 4 fusion dataset saved at {path}")
    return path


def ingest_phase3_gdsc_pediatric():
    """Create the Phase 3 pediatric drug-response evidence table using GDSC2 + pediatric-specific data."""
    response_dir = Path(CONFIG["raw_data_path"]) / "gdsc_pediatric"
    df = build_phase3_dataset(output_dir=response_dir)
    print(f"✅ GDSC2 + Pediatric Response Phase 3 dataset saved: {len(df)} rows at {response_dir}")
    return df


def ingest_phase4_mechanistic():
    """Create the Phase 4 mechanistic dataset linking drugs to targets and pathways."""
    mechanistic_dir = Path(CONFIG["raw_data_path"]) / "mechanistic"
    df = build_phase4_dataset(output_dir=mechanistic_dir)
    print(f"✅ ChEMBL + Reactome + STRING Phase 4 dataset saved: {len(df)} rows at {mechanistic_dir}")
    return df


def ingest_phase5_target_evidence():
    """Create the Phase 5 target-evidence dataset using BindingDB + GO + Open Targets."""
    target_evidence_dir = Path(CONFIG["raw_data_path"]) / "target_evidence"
    df = build_phase5_dataset(output_dir=target_evidence_dir)
    print(f"✅ BindingDB + GO + Open Targets Phase 5 dataset saved: {len(df)} rows at {target_evidence_dir}")
    return df


def ingest_phase6_structural():
    """Create the Phase 6 structural-evidence dataset using PDB interaction evidence."""
    structural_dir = Path(CONFIG["raw_data_path"]) / "structural"
    df = build_phase6_dataset(output_dir=structural_dir)
    print(f"✅ PDB/Structural Evidence Phase 6 dataset saved: {len(df)} rows at {structural_dir}")
    return df


def ingest_phase7_experimental():
    """Create the Phase 7 validation dataset summarizing experimental evidence."""
    experimental_dir = Path(CONFIG["raw_data_path"]) / "experimental"
    df = build_phase7_dataset(output_dir=experimental_dir)
    print(f"✅ Experimental Validation Phase 7 dataset saved: {len(df)} rows at {experimental_dir}")
    return df


if __name__ == "__main__":
    download_gdsc_data()
    ingest_chembl_target_evidence()
    ingest_phase1_openpedcan()
    ingest_phase2_openpedcan_mutations()
    ingest_phase3_openpedcan_copy_number()
    ingest_phase4_openpedcan_fusions()