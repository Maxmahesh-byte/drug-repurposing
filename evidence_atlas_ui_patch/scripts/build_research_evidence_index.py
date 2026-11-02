#!/usr/bin/env python3
"""Build compact, auditable dashboard indexes from existing project data.

Reads existing OpenPedCan, GDSC2 and ChEMBL CSVs. Does not modify source files,
Graph V2, Graph V3, or the intergenic evidence index.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]

SOURCES = {
    "clinical": ROOT / "data/raw/openpedcan/openpedcan_phase1.csv",
    "mutations": ROOT / "data/raw/openpedcan/openpedcan_mutations_phase2.csv",
    "gdsc": ROOT / "data/raw/gdsc/gdsc2_responses.csv",
    "chembl": ROOT / "data/raw/chembl/gdsc2_chembl_target_evidence.csv",
}
OUT_DIR = ROOT / "data/processed/graph_v3"
DB_PATH = OUT_DIR / "research_evidence.sqlite"
TEMP_DB = OUT_DIR / "research_evidence.sqlite.building"
MANIFEST_PATH = OUT_DIR / "research_evidence_index_manifest.json"
TEMP_MANIFEST = OUT_DIR / "research_evidence_index_manifest.json.tmp"
BATCH_SIZE = 100_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(reader: csv.DictReader, required: Iterable[str], source: str) -> None:
    absent = sorted(set(required) - set(reader.fieldnames or []))
    if absent:
        raise ValueError(f"{source} is missing required columns: {absent}")


def safe_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def flush_mutation_counts(connection: sqlite3.Connection, counts: Counter) -> None:
    if not counts:
        return
    connection.executemany(
        """
        INSERT INTO mutation_gene_summary
            (tumor_type, gene_symbol, variant_classification, observation_count)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(tumor_type, gene_symbol, variant_classification)
        DO UPDATE SET observation_count = observation_count + excluded.observation_count
        """,
        [(*key, value) for key, value in counts.items()],
    )
    connection.commit()
    counts.clear()


def main() -> None:
    missing = [str(path) for path in SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required source files are missing:\n" + "\n".join(missing))
    if DB_PATH.exists() or MANIFEST_PATH.exists() or TEMP_DB.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing index. Inspect/remove {DB_PATH}, "
            f"{MANIFEST_PATH}, or {TEMP_DB} before rebuilding."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: sha256_file(path) for name, path in SOURCES.items()}
    print("Source checksums recorded.")

    connection: sqlite3.Connection | None = None
    counts = {
        "clinical_rows": 0,
        "clinical_unique_samples": 0,
        "mutation_rows_scanned": 0,
        "gene_associated_variant_rows": 0,
        "unknown_intergenic_rows_excluded": 0,
        "gdsc_response_rows": 0,
        "chembl_evidence_rows": 0,
    }

    try:
        connection = sqlite3.connect(TEMP_DB)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=FILE")

        connection.executescript(
            """
            CREATE TABLE clinical_tumor_summary (
                tumor_type TEXT PRIMARY KEY,
                biospecimen_count INTEGER NOT NULL
            );

            CREATE TABLE mutation_gene_summary (
                tumor_type TEXT NOT NULL,
                gene_symbol TEXT NOT NULL,
                variant_classification TEXT NOT NULL,
                observation_count INTEGER NOT NULL,
                PRIMARY KEY (tumor_type, gene_symbol, variant_classification)
            );

            CREATE TABLE gdsc_response (
                row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                result_id TEXT,
                drug_id TEXT,
                drug_name TEXT NOT NULL,
                cell_line TEXT,
                sanger_model_id TEXT,
                reported_cancer_type TEXT,
                putative_target TEXT,
                pathway TEXT,
                ln_ic50 REAL,
                auc REAL,
                rmse REAL
            );

            CREATE TABLE chembl_target_evidence (
                row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                gdsc_drug_id TEXT,
                gdsc_drug_name TEXT,
                molecule_chembl_id TEXT,
                drug_match_method TEXT,
                matched_chembl_synonym TEXT,
                target_chembl_id TEXT,
                target_name TEXT,
                target_type TEXT,
                uniprot_accession TEXT,
                gene_symbol TEXT,
                mechanism_of_action TEXT,
                action_type TEXT,
                direct_interaction TEXT,
                max_phase TEXT,
                source_url TEXT
            );
            """
        )

        # 1) Clinical cohort summaries; the file is small enough to retain a
        # set of sample IDs per tumour category without loading all columns.
        print("[1/4] Summarising clinical cohort metadata...")
        clinical_samples: dict[str, set[str]] = defaultdict(set)
        with SOURCES["clinical"].open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            require_columns(reader, ["sample_id", "tumor_type"], "OpenPedCan clinical file")
            for row in reader:
                tumor = (row.get("tumor_type") or "").strip()
                sample = (row.get("sample_id") or "").strip()
                if tumor and sample:
                    clinical_samples[tumor].add(sample)
                counts["clinical_rows"] += 1
        connection.executemany(
            "INSERT INTO clinical_tumor_summary VALUES (?, ?)",
            [(tumor, len(samples)) for tumor, samples in clinical_samples.items()],
        )
        connection.commit()
        counts["clinical_unique_samples"] = len(
            set().union(*clinical_samples.values()) if clinical_samples else set()
        )

        # 2) Gene-associated mutation summaries. IGR/Unknown records are
        # deliberately excluded; they remain in their separate evidence index.
        print("[2/4] Streaming and aggregating gene-associated variants...")
        mutation_batch: Counter = Counter()
        with SOURCES["mutations"].open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            require_columns(
                reader,
                ["tumor_type", "gene_symbol", "variant_classification"],
                "OpenPedCan mutation file",
            )
            for row in reader:
                counts["mutation_rows_scanned"] += 1
                gene = (row.get("gene_symbol") or "").strip()
                tumor = (row.get("tumor_type") or "").strip()
                classification = (row.get("variant_classification") or "").strip()
                if gene.casefold() == "unknown" or classification.upper() == "IGR" or not gene:
                    counts["unknown_intergenic_rows_excluded"] += 1
                    continue
                if not tumor:
                    continue
                mutation_batch[(tumor, gene, classification or "Unspecified")] += 1
                counts["gene_associated_variant_rows"] += 1
                if counts["mutation_rows_scanned"] % BATCH_SIZE == 0:
                    flush_mutation_counts(connection, mutation_batch)
                    print(f"  scanned {counts['mutation_rows_scanned']:,} mutation rows...")
            flush_mutation_counts(connection, mutation_batch)

        # 3) GDSC2 response observations. Keep source measurements, not a
        # fabricated efficacy score. The dashboard will label these in vitro.
        print("[3/4] Indexing GDSC2 response observations...")
        gdsc_batch = []
        with SOURCES["gdsc"].open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            require_columns(
                reader,
                ["drug_id", "drug_name", "cell_line", "reported_cancer_type", "ln_ic50", "auc"],
                "GDSC2 response file",
            )
            for row in reader:
                gdsc_batch.append((
                    row.get("result_id"),
                    row.get("drug_id"),
                    (row.get("drug_name") or "").strip(),
                    row.get("cell_line"),
                    row.get("sanger_model_id"),
                    row.get("reported_cancer_type"),
                    row.get("putative_target"),
                    row.get("pathway"),
                    safe_float(row.get("ln_ic50")),
                    safe_float(row.get("auc")),
                    safe_float(row.get("rmse")),
                ))
                counts["gdsc_response_rows"] += 1
                if len(gdsc_batch) >= BATCH_SIZE:
                    connection.executemany(
                        """INSERT INTO gdsc_response
                        (result_id, drug_id, drug_name, cell_line, sanger_model_id,
                         reported_cancer_type, putative_target, pathway, ln_ic50, auc, rmse)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        gdsc_batch,
                    )
                    connection.commit()
                    gdsc_batch.clear()
            if gdsc_batch:
                connection.executemany(
                    """INSERT INTO gdsc_response
                    (result_id, drug_id, drug_name, cell_line, sanger_model_id,
                     reported_cancer_type, putative_target, pathway, ln_ic50, auc, rmse)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    gdsc_batch,
                )
                connection.commit()

        # 4) ChEMBL target evidence. Keep match method and source URL visible.
        print("[4/4] Indexing ChEMBL target evidence...")
        chembl_batch = []
        chembl_columns = [
            "gdsc_drug_id", "gdsc_drug_name", "molecule_chembl_id",
            "drug_match_method", "matched_chembl_synonym", "target_chembl_id",
            "target_name", "target_type", "uniprot_accession", "gene_symbol",
            "mechanism_of_action", "action_type", "direct_interaction",
            "max_phase", "source_url",
        ]
        with SOURCES["chembl"].open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            require_columns(reader, chembl_columns, "ChEMBL target evidence file")
            for row in reader:
                chembl_batch.append(tuple(row.get(column) for column in chembl_columns))
                counts["chembl_evidence_rows"] += 1
            connection.executemany(
                f"INSERT INTO chembl_target_evidence ({', '.join(chembl_columns)}) "
                f"VALUES ({', '.join('?' for _ in chembl_columns)})",
                chembl_batch,
            )
            connection.commit()

        print("Creating query indexes...")
        connection.executescript(
            """
            CREATE INDEX idx_mutation_disease_gene
                ON mutation_gene_summary(tumor_type, gene_symbol);
            CREATE INDEX idx_gdsc_drug
                ON gdsc_response(drug_id, drug_name);
            CREATE INDEX idx_gdsc_disease
                ON gdsc_response(reported_cancer_type);
            CREATE INDEX idx_chembl_drug_id
                ON chembl_target_evidence(gdsc_drug_id);
            CREATE INDEX idx_chembl_drug_name
                ON chembl_target_evidence(gdsc_drug_name);
            """
        )
        connection.commit()

        # Integrity and row-count checks before publishing.
        integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite quick_check failed: {integrity}")
        mutation_groups = connection.execute(
            "SELECT COUNT(*) FROM mutation_gene_summary"
        ).fetchone()[0]
        disease_count = connection.execute(
            "SELECT COUNT(*) FROM clinical_tumor_summary"
        ).fetchone()[0]
        gdsc_count = connection.execute(
            "SELECT COUNT(*) FROM gdsc_response"
        ).fetchone()[0]
        chembl_count = connection.execute(
            "SELECT COUNT(*) FROM chembl_target_evidence"
        ).fetchone()[0]
        connection.close()
        connection = None

        # Check that source files did not change during the build.
        final_hashes = {name: sha256_file(path) for name, path in SOURCES.items()}
        if final_hashes != source_hashes:
            raise RuntimeError("A source CSV changed while the index was being built.")

        manifest = {
            "schema_version": "1.0.0",
            "database": str(DB_PATH.relative_to(ROOT)),
            "source_hashes_sha256": source_hashes,
            "counts": counts,
            "summary_rows": {
                "clinical_tumour_types": disease_count,
                "mutation_tumour_gene_class_groups": mutation_groups,
                "gdsc_response_measurements": gdsc_count,
                "chembl_target_evidence_rows": chembl_count,
            },
            "mutation_policy": {
                "gene_symbol_unknown_excluded": True,
                "intergenic_igr_excluded_from_gene_overlap": True,
                "counts_represent_observations_not_unique_patients": True,
            },
            "response_policy": {
                "source": "GDSC2 in-vitro cell-line response measurements",
                "no_clinical_efficacy_score_generated": True,
            },
            "validation": {
                "source_files_unchanged_during_build": True,
                "sqlite_quick_check": True,
                "clinical_rows_indexed": counts["clinical_rows"] > 0,
                "mutation_rows_scanned": counts["mutation_rows_scanned"] > 0,
                "gdsc_rows_indexed": gdsc_count == counts["gdsc_response_rows"],
                "chembl_rows_indexed": chembl_count == counts["chembl_evidence_rows"],
            },
        }
        TEMP_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(TEMP_DB, DB_PATH)
        os.replace(TEMP_MANIFEST, MANIFEST_PATH)

        print("\nRESEARCH EVIDENCE INDEX COMPLETE")
        print("=" * 52)
        for key, value in counts.items():
            print(f"{key}: {value:,}")
        print(f"Mutation summary groups: {mutation_groups:,}")
        print(f"GDSC2 rows indexed: {gdsc_count:,}")
        print(f"ChEMBL rows indexed: {chembl_count:,}")
        print(f"SQLite integrity: {integrity}")
        print(f"Database: {DB_PATH}")
        print(f"Manifest: {MANIFEST_PATH}")
        print("Original source data and graph artifacts were not modified.")

    except Exception:
        if connection is not None:
            connection.close()
        TEMP_DB.unlink(missing_ok=True)
        Path(str(TEMP_DB) + "-journal").unlink(missing_ok=True)
        TEMP_MANIFEST.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
