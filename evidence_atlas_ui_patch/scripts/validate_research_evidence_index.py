#!/usr/bin/env python3
"""Read-only structural checks for the Evidence Atlas research index."""
import json
import sqlite3
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/processed/graph_v3/research_evidence.sqlite"
MANIFEST = ROOT / "data/processed/graph_v3/research_evidence_index_manifest.json"


def main():
    if not DB.is_file() or not MANIFEST.is_file():
        raise FileNotFoundError("Research index database or manifest is missing.")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    uri = "file:" + quote(str(DB), safe="/") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        integrity = con.execute("PRAGMA quick_check").fetchone()[0]
        counts = {
            "clinical_tumor_summary": con.execute("SELECT COUNT(*) FROM clinical_tumor_summary").fetchone()[0],
            "mutation_gene_summary": con.execute("SELECT COUNT(*) FROM mutation_gene_summary").fetchone()[0],
            "gdsc_response": con.execute("SELECT COUNT(*) FROM gdsc_response").fetchone()[0],
            "chembl_target_evidence": con.execute("SELECT COUNT(*) FROM chembl_target_evidence").fetchone()[0],
        }
        indexes = {
            row[0] for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
    finally:
        con.close()

    expected = manifest.get("summary_rows", {})
    checks = {
        "sqlite_quick_check": integrity == "ok",
        "clinical_summary_present": counts["clinical_tumor_summary"] > 0,
        "mutation_summary_present": counts["mutation_gene_summary"] > 0,
        "gdsc_rows_match_manifest": counts["gdsc_response"] == expected.get("gdsc_response_measurements"),
        "chembl_rows_match_manifest": counts["chembl_target_evidence"] == expected.get("chembl_target_evidence_rows"),
        "gdsc_drug_index_present": "idx_gdsc_drug" in indexes,
        "mutation_disease_gene_index_present": "idx_mutation_disease_gene" in indexes,
    }

    print("EVIDENCE ATLAS INDEX VALIDATION")
    print("=" * 45)
    print("SQLite integrity:", integrity)
    for name, value in counts.items():
        print(f"{name}: {value:,}")
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print("RESULT:", "PASS" if all(checks.values()) else "INVESTIGATE")

    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
