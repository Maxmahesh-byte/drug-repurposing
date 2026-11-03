#!/usr/bin/env python3
"""Read-only integrity and provenance tests for Evidence Engine v0.1."""

import csv
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

def main() -> int:
    ROOT = Path.cwd()
    DB = ROOT / "data/ai/evidence_engine_v0.sqlite"

    SOURCES = {
        "data/raw/gdsc/gdsc2_responses.csv":
            ("GDSC2_ROW_", "IN_VITRO_ASSAY_RESPONSE"),
        "data/raw/chembl/gdsc2_chembl_target_evidence.csv":
            ("CHEMBL_TARGET_ROW_", "DRUG_TARGET_ANNOTATION"),
    }

    checks = []

    def check(name, passed, detail=""):
        checks.append((name, bool(passed), detail))
        print(f"{'PASS' if passed else 'FAIL'} {name}"
              + (f" — {detail}" if detail else ""))


    def sha256(path):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()


    if not DB.exists():
        raise SystemExit(f"Database missing: {DB}")

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    check("sqlite_integrity", integrity == "ok", integrity)

    registry = {
        row["path"]: row
        for row in conn.execute("SELECT * FROM source_registry")
    }

    for relative, (prefix, expected_type) in SOURCES.items():
        path = ROOT / relative
        source = registry.get(relative)

        check(f"{prefix}source_registered", source is not None)
        if source is None or not path.exists():
            check(f"{prefix}source_file_available", False)
            continue

        check(f"{prefix}source_hash_matches",
              bool(source["sha256"]) and source["sha256"] == sha256(path))

        expected_count = 0
        exact_matches = 0
        unique_ids = set()
        all_rows_valid = True

        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for row_number, original in enumerate(reader, start=2):
                expected_count += 1
                record_id = f"{prefix}{row_number}"

                record = conn.execute("""
                    SELECT source_id, source_row_number, evidence_type,
                           raw_row_json, provenance_state
                    FROM evidence_records
                    WHERE record_id = ?
                """, (record_id,)).fetchone()

                if record is None:
                    all_rows_valid = False
                    continue

                if record["source_id"] != source["source_id"]:
                    all_rows_valid = False
                if record["source_row_number"] != row_number:
                    all_rows_valid = False
                if record["evidence_type"] != expected_type:
                    all_rows_valid = False

                try:
                    stored = json.loads(record["raw_row_json"])
                    if stored == original:
                        exact_matches += 1
                    else:
                        all_rows_valid = False
                except (TypeError, json.JSONDecodeError):
                    all_rows_valid = False

                if not record["provenance_state"]:
                    all_rows_valid = False

                unique_ids.add(record_id)

        actual_count = conn.execute("""
            SELECT COUNT(*) FROM evidence_records WHERE source_id = ?
        """, (source["source_id"],)).fetchone()[0]

        check(f"{prefix}row_count_matches",
              expected_count == actual_count,
              f"source={expected_count}, engine={actual_count}")

        check(f"{prefix}all_source_rows_preserved",
              all_rows_valid and exact_matches == expected_count,
              f"exact matches={exact_matches}/{expected_count}")

        check(f"{prefix}internal_ids_unique",
              len(unique_ids) == expected_count)

    # Ensure quarantined sources are excluded from normal evidence ingestion.
    quarantined = conn.execute("""
        SELECT DISTINCT s.source_id, s.path
        FROM source_registry s
        JOIN review_queue q ON q.source_id = s.source_id
    """).fetchall()

    contamination = []
    for source in quarantined:
        count = conn.execute("""
            SELECT COUNT(*) FROM evidence_records WHERE source_id = ?
        """, (source["source_id"],)).fetchone()[0]
        if count:
            contamination.append((source["path"], count))

    check("quarantined_sources_excluded",
          not contamination,
          f"contaminated sources={contamination}")

    review_count = conn.execute(
        "SELECT COUNT(*) FROM review_queue"
    ).fetchone()[0]
    check("review_queue_present", review_count > 0,
          f"quarantined rows={review_count}")

    # Diagnostic only: do not automatically interpret one result_id as corruption.
    gdsc = ROOT / "data/raw/gdsc/gdsc2_responses.csv"
    with gdsc.open(newline="", encoding="utf-8-sig") as f:
        result_ids = {
            (row.get("result_id") or "").strip()
            for row in csv.DictReader(f)
            if (row.get("result_id") or "").strip()
        }

    check("result_id_cardinality_reported",
          True, f"distinct nonblank result_id values={len(result_ids)}; semantics need review")

    conn.close()

    failed = [name for name, passed, _ in checks if not passed]
    print(f"\nTests passed: {len(checks) - len(failed)}/{len(checks)}")
    print("Overall:", "PASS" if not failed else "FAIL")
    return 1 if failed else 0



if __name__ == "__main__":
    raise SystemExit(main())
