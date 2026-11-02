#!/usr/bin/env python3
"""Build RePharma's provenance-first evidence store v0.

This is an ingestion and audit component, not a drug-response predictor.
It never joins drugs to tumour types to rank or recommend treatments.
"""

import csv
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
INVENTORY = ROOT / "data/ai/audits/molecular_feature_inventory.csv"
DB_PATH = ROOT / "data/ai/evidence_engine_v0.sqlite"
REPORT_PATH = ROOT / "data/ai/audits/evidence_engine_v0_report.json"

QUARANTINE = {
    "data/raw/experimental/experimental_validation_phase7.csv":
        "Small experimental-claim file lacks traceable assay/study identifiers.",
    "data/raw/mechanistic/chembl_reactome_string_phase4.csv":
        "Combined source label and scores are insufficient to verify each claim.",
    "data/raw/structural/pdb_structural_phase6.csv":
        "Structural mappings require structure-level verification; discrepancies flagged.",
    "data/raw/target/target_phase2.csv":
        "Sample identifiers and molecular assertions require source-level reconciliation.",
    "data/raw/target_evidence/bindingdb_go_opentargets_phase5.csv":
        "Small combined-source file lacks sufficient source-record identifiers.",
    "data/raw/gdsc_pediatric/gdsc_pediatric_phase3.csv":
        "Pediatric response rows require reconciliation to original assay records.",
    "data/raw/new_batch_2026.csv":
        "Unverified batch; contains unresolved entity names.",
    "data/raw/gdsc2_original.csv":
        "Small extract; source and completeness need verification.",
}

GDSC_PATH = "data/raw/gdsc/gdsc2_responses.csv"
CHEMBL_PATH = "data/raw/chembl/gdsc2_chembl_target_evidence.csv"
HASH_LIMIT = 64 * 1024 * 1024


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_source(path, status_from_inventory=""):
    if path in QUARANTINE:
        return (
            "QUARANTINED_UNVERIFIED",
            QUARANTINE[path],
        )

    if "gdsc_openpedcan_crosswalk" in path or \
       "gdsc_openpedcan_mapping_candidates_review" in path:
        return (
            "DRAFT_CROSSWALK_NOT_APPROVED",
            "Candidate identity mapping; not approved for automatic joins.",
        )

    if "graph_v3_prototype" in path or "graph_v2" in path:
        return (
            "DERIVED_OR_PROTOTYPE",
            "Derived/prototype artifact; trace to source before downstream use.",
        )

    if path == GDSC_PATH:
        return (
            "SOURCE_DATA_CATALOGUED",
            "Experimental cell-line assay data; not clinical efficacy evidence.",
        )

    if path == CHEMBL_PATH:
        return (
            "SOURCE_DATA_CATALOGUED",
            "Drug-target annotations; match and claim types require review.",
        )

    if path.startswith("data/raw/openpedcan/"):
        return (
            "SOURCE_DATA_CATALOGUED",
            "Molecular/clinical source data; original source and field semantics must be preserved.",
        )

    if path.startswith("data/raw/"):
        return (
            "SOURCE_REQUIRES_REVIEW",
            "Raw-directory file; provenance has not been independently established by this script.",
        )

    return (
        "DERIVED_OR_REQUIRES_REVIEW",
        "Processed artifact; inspect its source lineage before reuse.",
    )


def table_count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def main():
    if not (ROOT / "data").exists():
        raise SystemExit("Run from the project root containing data/.")

    if not INVENTORY.exists():
        raise SystemExit(
            "Missing molecular feature inventory. Run "
            "scripts/ai/audit_molecular_feature_inventory.py first."
        )

    with INVENTORY.open(newline="", encoding="utf-8-sig") as f:
        inventory = list(csv.DictReader(f))

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS source_registry (
            source_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            table_or_sheet TEXT NOT NULL DEFAULT '',
            evidence_category TEXT NOT NULL,
            trust_state TEXT NOT NULL,
            size_bytes INTEGER,
            observed_row_count TEXT,
            columns_json TEXT NOT NULL,
            sha256 TEXT,
            hash_state TEXT NOT NULL,
            rationale TEXT NOT NULL,
            registered_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS evidence_records (
            record_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL REFERENCES source_registry(source_id),
            source_row_number INTEGER NOT NULL,
            evidence_type TEXT NOT NULL,
            native_record_id TEXT,
            entity_ids_json TEXT NOT NULL,
            provenance_state TEXT NOT NULL,
            permitted_use TEXT NOT NULL,
            raw_row_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS review_queue (
            review_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL REFERENCES source_registry(source_id),
            source_row_number INTEGER NOT NULL,
            review_state TEXT NOT NULL,
            review_reason TEXT NOT NULL,
            raw_row_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS engine_checks (
            check_name TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            details_json TEXT NOT NULL,
            checked_at_utc TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_evidence_type
            ON evidence_records(evidence_type);
        CREATE INDEX IF NOT EXISTS idx_evidence_source
            ON evidence_records(source_id);
        CREATE INDEX IF NOT EXISTS idx_review_state
            ON review_queue(review_state);
    """)

    # Rebuild only the tables belonging to this generated v0 database.
    for table in (
        "evidence_records", "review_queue",
        "engine_checks", "source_registry"
    ):
        conn.execute(f"DELETE FROM {table}")

    source_by_path = {}
    hash_cache = {}
    trust_counts = Counter()

    for item in inventory:
        path_text = item["path"]
        table_name = item.get("table_or_sheet", "") or ""
        full_path = ROOT / path_text

        source_id = hashlib.sha256(
            f"{path_text}|{table_name}".encode()
        ).hexdigest()[:20]

        trust_state, rationale = classify_source(
            path_text, item.get("status", "")
        )
        trust_counts[trust_state] += 1

        size_bytes = (
            full_path.stat().st_size
            if full_path.exists() else None
        )

        file_hash = None
        hash_state = "NOT_COMPUTED"

        if not full_path.exists():
            hash_state = "FILE_MISSING"
        elif size_bytes <= HASH_LIMIT:
            if path_text not in hash_cache:
                hash_cache[path_text] = sha256_file(full_path)
            file_hash = hash_cache[path_text]
            hash_state = "SHA256_COMPUTED"
        else:
            hash_state = "SKIPPED_LARGE_FILE"

        conn.execute("""
            INSERT INTO source_registry (
                source_id, path, table_or_sheet, evidence_category,
                trust_state, size_bytes, observed_row_count, columns_json,
                sha256, hash_state, rationale, registered_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            source_id,
            path_text,
            table_name,
            item.get("evidence_category", "unknown"),
            trust_state,
            size_bytes,
            item.get("row_count", ""),
            json.dumps([
                c.strip()
                for c in item.get("columns", "").split("|")
                if c.strip()
            ]),
            file_hash,
            hash_state,
            rationale,
            now(),
        ))

        source_by_path.setdefault(path_text, source_id)

    # Import raw GDSC2 assay records as experimental observations.
    gdsc_stats = {
        "rows_read": 0,
        "missing_curve_id": 0,
        "duplicate_curve_id_rows": 0,
        "unique_result_ids": 0,
        "missing_drug_id": 0,
        "missing_model_id": 0,
    }

    gdsc_path = ROOT / GDSC_PATH
    if not gdsc_path.exists():
        raise SystemExit(f"Required source missing: {GDSC_PATH}")

    curve_counts = Counter()
    result_ids = set()
    gdsc_rows = []

    with gdsc_path.open(newline="", encoding="utf-8-sig") as f:
        for row_number, row in enumerate(csv.DictReader(f), start=2):
            gdsc_rows.append((row_number, row))
            gdsc_stats["rows_read"] += 1

            curve = (row.get("curve_id") or "").strip()
            if not curve:
                gdsc_stats["missing_curve_id"] += 1
            else:
                curve_counts[curve] += 1

            result_id = (row.get("result_id") or "").strip()
            if result_id:
                result_ids.add(result_id)

            if not (row.get("drug_id") or "").strip():
                gdsc_stats["missing_drug_id"] += 1
            if not (row.get("sanger_model_id") or "").strip():
                gdsc_stats["missing_model_id"] += 1

    gdsc_stats["unique_result_ids"] = len(result_ids)
    gdsc_stats["duplicate_curve_id_rows"] = sum(
        count - 1 for count in curve_counts.values() if count > 1
    )

    gdsc_source_id = source_by_path[GDSC_PATH]

    for row_number, row in gdsc_rows:
        curve = (row.get("curve_id") or "").strip()
        drug_id = (row.get("drug_id") or "").strip()
        sanger_id = (row.get("sanger_model_id") or "").strip()
        cosmic_id = (row.get("cosmic_id") or "").strip()

        # Internal record IDs use row numbers to preserve every source row,
        # even if a source-native identifier is duplicated.
        record_id = f"GDSC2_ROW_{row_number}"

        entity_ids = {
            "drug": f"GDSC_DRUG:{drug_id}" if drug_id else None,
            "cell_model": f"SANGER_MODEL:{sanger_id}" if sanger_id else None,
            "cosmic_model": f"COSMIC:{cosmic_id}" if cosmic_id else None,
        }

        if not curve:
            provenance = "MISSING_NATIVE_CURVE_ID_REVIEW_REQUIRED"
        elif curve_counts[curve] > 1:
            provenance = "DUPLICATE_NATIVE_CURVE_ID_REVIEW_REQUIRED"
        else:
            provenance = "SOURCE_FILE_AND_CURVE_ID_PRESENT"

        conn.execute("""
            INSERT INTO evidence_records (
                record_id, source_id, source_row_number, evidence_type,
                native_record_id, entity_ids_json, provenance_state,
                permitted_use, raw_row_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record_id,
            gdsc_source_id,
            row_number,
            "IN_VITRO_ASSAY_RESPONSE",
            curve or None,
            json.dumps(entity_ids),
            provenance,
            "ASSAY_CONTEXT_ONLY_NOT_CLINICAL_EFFICACY",
            json.dumps(row, sort_keys=True),
        ))

    # Import ChEMBL target annotations separately; do not reinterpret them
    # as measured binding or validated disease-specific effects.
    chembl_stats = {
        "rows_read": 0,
        "missing_source_url": 0,
        "missing_match_method": 0,
        "missing_molecule_chembl_id": 0,
        "missing_target_chembl_id": 0,
    }

    chembl_path = ROOT / CHEMBL_PATH
    if not chembl_path.exists():
        raise SystemExit(f"Required source missing: {CHEMBL_PATH}")

    chembl_source_id = source_by_path[CHEMBL_PATH]

    with chembl_path.open(newline="", encoding="utf-8-sig") as f:
        for row_number, row in enumerate(csv.DictReader(f), start=2):
            chembl_stats["rows_read"] += 1

            source_url = (row.get("source_url") or "").strip()
            match_method = (row.get("drug_match_method") or "").strip()
            molecule = (row.get("molecule_chembl_id") or "").strip()
            target = (row.get("target_chembl_id") or "").strip()

            if not source_url:
                chembl_stats["missing_source_url"] += 1
            if not match_method:
                chembl_stats["missing_match_method"] += 1
            if not molecule:
                chembl_stats["missing_molecule_chembl_id"] += 1
            if not target:
                chembl_stats["missing_target_chembl_id"] += 1

            gdsc_id = (row.get("gdsc_drug_id") or "").strip()
            uniprot = (row.get("uniprot_accession") or "").strip()
            gene = (row.get("gene_symbol") or "").strip()

            entity_ids = {
                "gdsc_drug": f"GDSC_DRUG:{gdsc_id}" if gdsc_id else None,
                "chembl_molecule": molecule or None,
                "chembl_target": target or None,
                "uniprot": uniprot or None,
                "gene_symbol_as_recorded": gene or None,
            }

            if source_url and match_method:
                provenance = "SOURCE_URL_AND_MATCH_METHOD_RECORDED"
            else:
                provenance = "SOURCE_OR_MATCH_METADATA_MISSING_REVIEW"

            conn.execute("""
                INSERT INTO evidence_records (
                    record_id, source_id, source_row_number, evidence_type,
                    native_record_id, entity_ids_json, provenance_state,
                    permitted_use, raw_row_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"CHEMBL_TARGET_ROW_{row_number}",
                chembl_source_id,
                row_number,
                "DRUG_TARGET_ANNOTATION",
                source_url or None,
                json.dumps(entity_ids),
                provenance,
                "TARGET_ANNOTATION_ONLY_NOT_BINDING_EVIDENCE_BY_DEFAULT",
                json.dumps(row, sort_keys=True),
            ))

    # Quarantine unverified small/prototype files for manual review.
    review_stats = Counter()

    for relative_path, reason in QUARANTINE.items():
        path = ROOT / relative_path
        if not path.exists():
            review_stats["quarantined_files_missing"] += 1
            continue

        source_id = source_by_path.get(relative_path)
        if not source_id:
            review_stats["quarantined_files_not_in_inventory"] += 1
            continue

        with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
            for row_number, row in enumerate(csv.DictReader(f), start=2):
                review_id = hashlib.sha256(
                    f"{relative_path}|{row_number}".encode()
                ).hexdigest()[:24]

                conn.execute("""
                    INSERT INTO review_queue (
                        review_id, source_id, source_row_number,
                        review_state, review_reason, raw_row_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    review_id,
                    source_id,
                    row_number,
                    "QUARANTINED_PENDING_SOURCE_VERIFICATION",
                    reason,
                    json.dumps(row, sort_keys=True),
                ))
                review_stats["rows_quarantined"] += 1

        review_stats["files_quarantined"] += 1

    # Integrity/coverage checks; none of these constitute biological validation.
    checks = {
        "gdsc_curve_id_missing": {
            "status": "PASS" if gdsc_stats["missing_curve_id"] == 0 else "REVIEW",
            "value": gdsc_stats["missing_curve_id"],
        },
        "gdsc_curve_id_duplicates": {
            "status": "PASS" if gdsc_stats["duplicate_curve_id_rows"] == 0 else "REVIEW",
            "value": gdsc_stats["duplicate_curve_id_rows"],
        },
        "gdsc_result_id_cardinality": {
            "status": "REVIEW" if gdsc_stats["unique_result_ids"] == 1 else "INFO",
            "value": gdsc_stats["unique_result_ids"],
            "note": "In this GDSC2 release, NLME_RESULT_ID is shared across "
                "curves; NLME_CURVE_ID is unique per curve. "
                "Missing and duplicate curve IDs are checked separately.",
        },
        "chembl_source_url_coverage": {
            "status": "PASS" if chembl_stats["missing_source_url"] == 0 else "REVIEW",
            "missing": chembl_stats["missing_source_url"],
        },
        "chembl_match_method_coverage": {
            "status": "PASS" if chembl_stats["missing_match_method"] == 0 else "REVIEW",
            "missing": chembl_stats["missing_match_method"],
        },
        "unverified_records_quarantined": {
            "status": "INFO",
            **dict(review_stats),
        },
        "no_prediction_or_ranking_performed": {
            "status": "PASS",
            "value": True,
        },
    }

    for name, details in checks.items():
        conn.execute("""
            INSERT INTO engine_checks (
                check_name, status, details_json, checked_at_utc
            ) VALUES (?, ?, ?, ?)
        """, (
            name,
            details["status"],
            json.dumps(details, sort_keys=True),
            now(),
        ))

    conn.commit()

    report = {
        "engine_version": "0.1",
        "created_at_utc": now(),
        "database": str(DB_PATH.relative_to(ROOT)),
        "source_inventory_entries": len(inventory),
        "source_states": dict(trust_counts),
        "gdsc2_ingestion": gdsc_stats,
        "chembl_annotation_ingestion": chembl_stats,
        "review_queue": dict(review_stats),
        "database_counts": {
            "source_registry": table_count(conn, "source_registry"),
            "evidence_records": table_count(conn, "evidence_records"),
            "review_queue": table_count(conn, "review_queue"),
            "engine_checks": table_count(conn, "engine_checks"),
        },
        "checks": checks,
        "limitations": [
            "Source presence and identifiers do not prove each scientific claim.",
            "GDSC2 responses are in-vitro assay observations, not clinical outcomes.",
            "ChEMBL rows are annotations and do not automatically establish binding.",
            "Candidate crosswalks remain unapproved.",
            "Small phase files remain quarantined until verified against original records.",
            "No datasets were joined to rank drug effects for a tumour type.",
            "No model was trained and no efficacy predictions were made.",
        ],
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    conn.close()

    print("REPHARMA EVIDENCE ENGINE v0.1")
    print(f"Source inventory entries: {len(inventory):,}")
    print(f"GDSC2 response rows:       {gdsc_stats['rows_read']:,}")
    print(f"ChEMBL annotation rows:     {chembl_stats['rows_read']:,}")
    print(f"Quarantined rows:           {review_stats['rows_quarantined']:,}")
    print(f"Evidence records:           {report['database_counts']['evidence_records']:,}")
    print(f"Review queue rows:          {report['database_counts']['review_queue']:,}")
    print("\nChecks:")
    for name, details in checks.items():
        print(f"  {details['status']:6} {name}: {details}")
    print(f"\nDatabase: {DB_PATH.relative_to(ROOT)}")
    print(f"Report:   {REPORT_PATH.relative_to(ROOT)}")
    print("\nNo prediction or drug-ranking model was run.")


if __name__ == "__main__":
    main()
