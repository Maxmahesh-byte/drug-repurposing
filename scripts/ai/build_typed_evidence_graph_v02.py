#!/usr/bin/env python3
"""Extend Evidence Graph v0.1 with source-row-backed OpenPedCan evidence."""

import csv
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
BASE = ROOT / "data/ai/typed_evidence_graph_v0.sqlite"
OUT = ROOT / "data/ai/typed_evidence_graph_v0_2.sqlite"
REPORT = ROOT / "data/ai/audits/typed_evidence_graph_v0_2_report.json"

SOURCES = [
    {
        "prefix": "MUT",
        "layer": "SOMATIC_VARIANT",
        "path": "data/raw/openpedcan/openpedcan_mutations_phase2.csv",
        "delimiter": ",",
    },
    {
        "prefix": "CNV",
        "layer": "COPY_NUMBER",
        "path": "data/raw/openpedcan/openpedcan_copy_number_phase3.csv",
        "delimiter": ",",
    },
    {
        "prefix": "FUS",
        "layer": "GENE_FUSION",
        "path": "data/raw/openpedcan/openpedcan_fusions_phase4.csv",
        "delimiter": ",",
    },
]

OBS_FIELDS = [
    "observation_id", "source_id", "source_path", "source_row_number",
    "layer", "sample_id", "feature_node_id", "metadata_match_state",
    "gene_symbol", "variant_classification", "variant_type", "chromosome",
    "start_position", "reference_allele", "tumor_allele", "protein_change",
    "impact", "status", "copy_number", "ploidy", "cytoband",
    "fusion_name", "gene_a", "gene_b", "ensembl_a", "ensembl_b",
    "fusion_annotation", "fusion_type", "cohort", "tumor_type",
    "diagnosis", "source_label", "row_sha256",
]

def now():
    return datetime.now(timezone.utc).isoformat()

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def stable_hash(*parts):
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def exact_ids(path, column, delimiter):
    result = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter=delimiter):
            value = (row.get(column) or "").strip()
            if value:
                result.add(value)
    return result

def main():
    if not BASE.exists():
        raise SystemExit(f"Validated base graph missing: {BASE}")
    if OUT.exists():
        raise SystemExit(
            f"Output already exists: {OUT}\n"
            "Refusing to overwrite it. Inspect the existing file first."
        )

    for source in SOURCES:
        if not (ROOT / source["path"]).exists():
            raise SystemExit(f"Required source missing: {source['path']}")

    # Establish metadata match states using exact identifiers after trimming
    # surrounding whitespace. This does not assert diagnosis equivalence.
    clinical = exact_ids(
        ROOT / "data/raw/openpedcan/openpedcan_phase1.csv",
        "sample_id", ",",
    )
    histology = exact_ids(
        ROOT / "data/raw/openpedcan/v15/histologies.tsv",
        "sample_id", "\t",
    )

    print("Preparing v0.2 from the validated v0.1 graph...", flush=True)

    src = sqlite3.connect(f"file:{BASE}?mode=ro", uri=True)
    graph = sqlite3.connect(OUT)
    src.backup(graph)
    src.close()

    graph.execute("PRAGMA foreign_keys = ON")
    graph.execute("PRAGMA journal_mode = WAL")
    graph.execute("PRAGMA synchronous = NORMAL")

    # Resolve the already-registered source nodes and record the content hash
    # of each canonical phase export.
    source_nodes = {}
    for n in graph.execute("""
        SELECT node_id, attributes_json FROM graph_nodes
        WHERE node_type = 'SOURCE_FILE'
    """):
        attrs = json.loads(n["attributes_json"]) if isinstance(n, sqlite3.Row) else json.loads(n[1])
        source_nodes[attrs.get("path")] = {
            "node_id": n["node_id"] if isinstance(n, sqlite3.Row) else n[0],
            "source_id": attrs.get("source_id"),
        }

    source_hashes = {}
    for spec in SOURCES:
        if spec["path"] not in source_nodes or not source_nodes[spec["path"]]["source_id"]:
            graph.close()
            raise SystemExit(
                f"Source is not registered in v0.1: {spec['path']}. "
                "No molecular rows have been imported."
            )
        print(f"Hashing {spec['path']}...", flush=True)
        source_hashes[spec["path"]] = sha256_file(ROOT / spec["path"])

    graph.executescript("""
        CREATE TABLE molecular_sources (
            source_id TEXT PRIMARY KEY,
            source_path TEXT NOT NULL UNIQUE,
            source_sha256 TEXT NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            row_count INTEGER NOT NULL DEFAULT 0,
            imported_at_utc TEXT NOT NULL
        );

        CREATE TABLE molecular_observations (
            observation_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL REFERENCES molecular_sources(source_id),
            source_path TEXT NOT NULL,
            source_row_number INTEGER NOT NULL,
            layer TEXT NOT NULL,
            sample_id TEXT,
            feature_node_id TEXT NOT NULL,
            metadata_match_state TEXT NOT NULL,
            gene_symbol TEXT,
            variant_classification TEXT,
            variant_type TEXT,
            chromosome TEXT,
            start_position TEXT,
            reference_allele TEXT,
            tumor_allele TEXT,
            protein_change TEXT,
            impact TEXT,
            status TEXT,
            copy_number TEXT,
            ploidy TEXT,
            cytoband TEXT,
            fusion_name TEXT,
            gene_a TEXT,
            gene_b TEXT,
            ensembl_a TEXT,
            ensembl_b TEXT,
            fusion_annotation TEXT,
            fusion_type TEXT,
            cohort TEXT,
            tumor_type TEXT,
            diagnosis TEXT,
            source_label TEXT,
            row_sha256 TEXT NOT NULL,
            UNIQUE(source_id, source_row_number)
        );

        CREATE TABLE molecular_import_checks (
            check_name TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            details_json TEXT NOT NULL,
            checked_at_utc TEXT NOT NULL
        );
    """)

    for spec in SOURCES:
        source_id = source_nodes[spec["path"]]["source_id"]
        path = ROOT / spec["path"]
        graph.execute("""
            INSERT INTO molecular_sources
            (source_id, source_path, source_sha256, file_size_bytes, row_count,
             imported_at_utc)
            VALUES (?, ?, ?, ?, 0, ?)
        """, (
            source_id, spec["path"], source_hashes[spec["path"]],
            path.stat().st_size, now(),
        ))

    known_nodes = {
        row[0] for row in graph.execute("SELECT node_id FROM graph_nodes")
    }
    feature_counts = Counter()
    layer_counts = Counter()
    row_counts = Counter()
    metadata_states = Counter()
    unassigned_counts = Counter()
    orphan_sample_ids = {s["prefix"]: set() for s in SOURCES}
    unique_samples = {s["prefix"]: set() for s in SOURCES}
    unique_features = {s["prefix"]: set() for s in SOURCES}

    def add_node(node_id, node_type, label, attrs, trust):
        if node_id in known_nodes:
            return
        graph.execute("""
            INSERT INTO graph_nodes
            (node_id, node_type, display_label, attributes_json, trust_state)
            VALUES (?, ?, ?, ?, ?)
        """, (
            node_id, node_type, str(label),
            json.dumps(attrs, sort_keys=True, ensure_ascii=False),
            trust,
        ))
        known_nodes.add(node_id)
        feature_counts[node_type] += 1

    # Explicit unresolved-feature nodes preserve observations whose gene
    # assignment is blank or is a generic placeholder.
    for node_id, label, node_type in (
        ("FEATURE:OPENPEDCAN:UNASSIGNED_VARIANT",
         "Variant without a usable gene assignment", "UNASSIGNED_VARIANT_FEATURE"),
        ("FEATURE:OPENPEDCAN:UNASSIGNED_COPY_NUMBER",
         "Copy-number observation without a usable gene assignment",
         "UNASSIGNED_COPY_NUMBER_FEATURE"),
        ("FEATURE:OPENPEDCAN:UNASSIGNED_FUSION",
         "Fusion observation without a usable fusion signature",
         "UNASSIGNED_FUSION_FEATURE"),
    ):
        add_node(node_id, node_type, label, {"mapping": "unresolved"},
                 "UNASSIGNED_NOT_A_BIOLOGICAL_ENTITY")

    obs_sql = (
        "INSERT INTO molecular_observations ("
        + ", ".join(OBS_FIELDS)
        + ") VALUES ("
        + ", ".join("?" for _ in OBS_FIELDS)
        + ")"
    )
    edge_sql = """
        INSERT INTO graph_edges
        (edge_id, source_node_id, edge_type, target_node_id,
         evidence_record_id, assertion_status, provenance_state,
         source_row_number)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """

    # Each source row is imported exactly once from the canonical phase CSV.
    # The upstream v15 copies are deliberately not re-imported.
    for spec in SOURCES:
        prefix, layer = spec["prefix"], spec["layer"]
        relative = spec["path"]
        path = ROOT / relative
        source_id = source_nodes[relative]["source_id"]

        print(f"\nImporting {layer}: {relative}", flush=True)

        rows_batch = []
        edges_batch = []
        rows_seen = 0
        rows_without_sample = 0
        rows_missing_gene_or_signature = 0
        per_sample_edges = 0
        fusion_partners_seen = set()

        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=spec["delimiter"])
            columns = set(reader.fieldnames or [])
            if "sample_id" not in columns:
                raise SystemExit(f"sample_id column missing in {relative}")

            if layer == "SOMATIC_VARIANT" and "gene_symbol" not in columns:
                raise SystemExit(f"gene_symbol missing in {relative}")
            if layer == "COPY_NUMBER" and "gene_symbol" not in columns:
                raise SystemExit(f"gene_symbol missing in {relative}")
            if layer == "GENE_FUSION" and "fusion_name" not in columns:
                raise SystemExit(f"fusion_name missing in {relative}")

            for row_number, row in enumerate(reader, start=2):
                rows_seen += 1
                sample = (row.get("sample_id") or "").strip()
                raw_gene = row.get("gene_symbol") or ""
                gene = raw_gene.strip()
                row_hash = stable_hash(row)

                if not sample:
                    rows_without_sample += 1
                    metadata_state = "MISSING_SAMPLE_ID"
                    sample_node = None
                else:
                    in_clinical = sample in clinical
                    in_histology = sample in histology
                    if in_clinical and in_histology:
                        metadata_state = "MATCHES_CLINICAL_AND_HISTOLOGY"
                    elif in_clinical:
                        metadata_state = "MATCHES_CLINICAL_ONLY"
                    elif in_histology:
                        metadata_state = "MATCHES_HISTOLOGY_ONLY"
                    else:
                        metadata_state = "MATCHES_NEITHER_METADATA_TABLE"
                        orphan_sample_ids[prefix].add(sample)

                    metadata_states[metadata_state] += 1
                    unique_samples[prefix].add(sample)
                    sample_node = f"OPENPEDCAN_SAMPLE:{sample}"
                    add_node(
                        sample_node, "OPENPEDCAN_SAMPLE", sample,
                        {"sample_id_as_recorded": sample,
                         "clinical_metadata_match": metadata_state},
                        "SOURCE_IDENTIFIER_MATCH_ONLY",
                    )

                if layer == "SOMATIC_VARIANT":
                    if gene and gene.casefold() not in {"unknown", "na", "n/a", "."}:
                        feature_node = f"GENE_SYMBOL:OPENPEDCAN:{gene}"
                        add_node(
                            feature_node, "OPENPEDCAN_GENE_SYMBOL", gene,
                            {"gene_symbol_as_recorded": gene,
                             "normalization": "NOT_PERFORMED"},
                            "SOURCE_RECORDED_SYMBOL_NOT_NORMALIZED",
                        )
                        edge_type = "SAMPLE_HAS_MUTATION_OBSERVATION"
                    else:
                        feature_node = "FEATURE:OPENPEDCAN:UNASSIGNED_VARIANT"
                        edge_type = "SAMPLE_HAS_UNASSIGNED_VARIANT_OBSERVATION"
                        rows_missing_gene_or_signature += 1

                elif layer == "COPY_NUMBER":
                    if gene and gene.casefold() not in {"unknown", "na", "n/a", "."}:
                        feature_node = f"GENE_SYMBOL:OPENPEDCAN:{gene}"
                        add_node(
                            feature_node, "OPENPEDCAN_GENE_SYMBOL", gene,
                            {"gene_symbol_as_recorded": gene,
                             "normalization": "NOT_PERFORMED"},
                            "SOURCE_RECORDED_SYMBOL_NOT_NORMALIZED",
                        )
                        edge_type = "SAMPLE_HAS_COPY_NUMBER_OBSERVATION"
                    else:
                        feature_node = "FEATURE:OPENPEDCAN:UNASSIGNED_COPY_NUMBER"
                        edge_type = "SAMPLE_HAS_UNASSIGNED_COPY_NUMBER_OBSERVATION"
                        rows_missing_gene_or_signature += 1

                else:
                    fusion_name = (row.get("fusion_name") or "").strip()
                    gene_a = (row.get("gene_a") or "").strip()
                    gene_b = (row.get("gene_b") or "").strip()
                    ensembl_a = (row.get("ensembl_a") or "").strip()
                    ensembl_b = (row.get("ensembl_b") or "").strip()

                    signature_fields = [
                        fusion_name, gene_a, gene_b, ensembl_a, ensembl_b
                    ]
                    if any(signature_fields):
                        fusion_hash = stable_hash(*signature_fields)[:24]
                        feature_node = f"FUSION_SIGNATURE:OPENPEDCAN:{fusion_hash}"
                        add_node(
                            feature_node, "FUSION_SIGNATURE",
                            fusion_name or f"{gene_a}--{gene_b}",
                            {
                                "fusion_name_as_recorded": fusion_name,
                                "gene_a_as_recorded": gene_a,
                                "gene_b_as_recorded": gene_b,
                                "ensembl_a_as_recorded": ensembl_a,
                                "ensembl_b_as_recorded": ensembl_b,
                                "normalization": "NOT_PERFORMED",
                            },
                            "SOURCE_REPORTED_FUSION_SIGNATURE",
                        )
                        if feature_node not in fusion_partners_seen:
                            fusion_partners_seen.add(feature_node)
                            for partner_label, partner in (
                                ("A", gene_a), ("B", gene_b)
                            ):
                                if partner:
                                    partner_node = f"GENE_SYMBOL:OPENPEDCAN:{partner}"
                                    add_node(
                                        partner_node, "OPENPEDCAN_GENE_SYMBOL",
                                        partner,
                                        {"gene_symbol_as_recorded": partner,
                                         "normalization": "NOT_PERFORMED"},
                                        "SOURCE_RECORDED_SYMBOL_NOT_NORMALIZED",
                                    )
                                    edges_batch.append((
                                        "OPC_FUSION_PARTNER_" +
                                        stable_hash(feature_node, partner_label)[:28],
                                        feature_node,
                                        f"FUSION_REPORTED_PARTNER_{partner_label}",
                                        partner_node,
                                        None,
                                        "SOURCE_ANNOTATION",
                                        "SOURCE_REPORTED_FUSION_SIGNATURE",
                                        None,
                                    ))
                        edge_type = "SAMPLE_HAS_FUSION_OBSERVATION"
                    else:
                        feature_node = "FEATURE:OPENPEDCAN:UNASSIGNED_FUSION"
                        edge_type = "SAMPLE_HAS_UNASSIGNED_FUSION_OBSERVATION"
                        rows_missing_gene_or_signature += 1

                feature_id = feature_node
                unique_features[prefix].add(feature_id)
                observation_id = f"OPCP_{prefix}_ROW_{row_number}"

                values = {
                    "observation_id": observation_id,
                    "source_id": source_id,
                    "source_path": relative,
                    "source_row_number": row_number,
                    "layer": layer,
                    "sample_id": sample,
                    "feature_node_id": feature_id,
                    "metadata_match_state": metadata_state,
                    "gene_symbol": row.get("gene_symbol", "") or "",
                    "variant_classification": row.get("variant_classification", "") or "",
                    "variant_type": row.get("variant_type", "") or "",
                    "chromosome": row.get("chromosome", "") or "",
                    "start_position": row.get("start_position", "") or "",
                    "reference_allele": row.get("reference_allele", "") or "",
                    "tumor_allele": row.get("tumor_allele", "") or "",
                    "protein_change": row.get("protein_change", "") or "",
                    "impact": row.get("impact", "") or "",
                    "status": row.get("status", "") or "",
                    "copy_number": row.get("copy_number", "") or "",
                    "ploidy": row.get("ploidy", "") or "",
                    "cytoband": row.get("cytoband", "") or "",
                    "fusion_name": row.get("fusion_name", "") or "",
                    "gene_a": row.get("gene_a", "") or "",
                    "gene_b": row.get("gene_b", "") or "",
                    "ensembl_a": row.get("ensembl_a", "") or "",
                    "ensembl_b": row.get("ensembl_b", "") or "",
                    "fusion_annotation": row.get("fusion_annotation", "") or "",
                    "fusion_type": row.get("fusion_type", "") or "",
                    "cohort": row.get("cohort", "") or "",
                    "tumor_type": row.get("tumor_type", "") or "",
                    "diagnosis": row.get("diagnosis", "") or "",
                    "source_label": row.get("source", "") or "",
                    "row_sha256": row_hash,
                }
                rows_batch.append(tuple(values[k] for k in OBS_FIELDS))

                # The row-level molecular_observations table holds the
                # detailed attributes; this edge links the sample to the
                # recorded feature with an exact source-row pointer.
                if sample_node:
                    edges_batch.append((
                        f"OPC_EDGE_{prefix}_{row_number}",
                        sample_node,
                        edge_type,
                        feature_id,
                        observation_id,
                        "SOURCE_REPORTED_MOLECULAR_OBSERVATION",
                        "SOURCE_ROW_HASHED_OPENPEDCAN_EXPORT",
                        row_number,
                    ))
                    per_sample_edges += 1

                if len(rows_batch) >= 20_000:
                    graph.executemany(obs_sql, rows_batch)
                    rows_batch.clear()

                if len(edges_batch) >= 20_000:
                    graph.executemany(edge_sql, edges_batch)
                    edges_batch.clear()

                if rows_seen % 100_000 == 0:
                    graph.commit()
                    print(f"  {rows_seen:,} source rows processed", flush=True)

        if rows_batch:
            graph.executemany(obs_sql, rows_batch)
        if edges_batch:
            graph.executemany(edge_sql, edges_batch)

        graph.execute("""
            UPDATE molecular_sources SET row_count = ?
            WHERE source_id = ?
        """, (rows_seen, source_id))
        graph.commit()

        layer_counts[layer] = rows_seen
        row_counts[prefix] = {
            "rows": rows_seen,
            "rows_without_sample_id": rows_without_sample,
            "rows_without_gene_or_fusion_signature": rows_missing_gene_or_signature,
            "sample_feature_edges": per_sample_edges,
            "unique_sample_ids": len(unique_samples[prefix]),
            "unique_feature_nodes": len(unique_features[prefix]),
            "sample_ids_missing_from_both_metadata_tables":
                len(orphan_sample_ids[prefix]),
            "source_sha256": source_hashes[relative],
        }

        print(
            f"  Imported {rows_seen:,} rows; "
            f"{len(unique_samples[prefix]):,} unique sample IDs; "
            f"{len(orphan_sample_ids[prefix]):,} sample IDs absent from both metadata tables.",
            flush=True,
        )

    # Indexes are created after bulk ingest for more efficient construction.
    graph.execute("""
        CREATE INDEX idx_molecular_observations_sample_layer
        ON molecular_observations(sample_id, layer)
    """)
    graph.execute("""
        CREATE INDEX idx_molecular_observations_feature
        ON molecular_observations(feature_node_id)
    """)
    graph.execute("""
        CREATE INDEX idx_molecular_observations_source_row
        ON molecular_observations(source_id, source_row_number)
    """)

    # Base and molecular evidence remain separate tables; the report records
    # the row counts and verifies that each added observation has a graph edge.
    graph_integrity = graph.execute("PRAGMA integrity_check").fetchone()[0]
    fk_errors = graph.execute("PRAGMA foreign_key_check").fetchall()
    observed = graph.execute(
        "SELECT COUNT(*) FROM molecular_observations"
    ).fetchone()[0]
    observation_edges = graph.execute("""
        SELECT COUNT(*) FROM graph_edges
        WHERE evidence_record_id LIKE 'OPCP\\_%\\_ROW\\_%' ESCAPE '\\'
    """).fetchone()[0]
    missing_edges = graph.execute("""
        SELECT COUNT(*)
        FROM molecular_observations m
        WHERE m.sample_id != ''
          AND NOT EXISTS (
            SELECT 1 FROM graph_edges e
            WHERE e.evidence_record_id = m.observation_id
          )
    """).fetchone()[0]

    checks = {
        "sqlite_integrity": {
            "status": "PASS" if graph_integrity == "ok" else "FAIL",
            "details": graph_integrity,
        },
        "foreign_key_integrity": {
            "status": "PASS" if not fk_errors else "FAIL",
            "details": len(fk_errors),
        },
        "molecular_row_count_matches_import": {
            "status": "PASS" if observed == sum(layer_counts.values()) else "FAIL",
            "details": {"database": observed, "imported": sum(layer_counts.values())},
        },
        "molecular_observation_edges_present": {
            "status": "PASS" if missing_edges == 0 else "FAIL",
            "details": {
                "edges_with_observation_ids": observation_edges,
                "observations_missing_edges": missing_edges,
            },
        },
        "source_export_hashes_recorded": {
            "status": "PASS" if all(row.get("source_sha256") for row in row_counts.values()) else "FAIL",
            "details": "SHA-256 recorded for each canonical phase export.",
        },
        "no_sample_ids_missing_from_both_metadata_tables": {
            "status": "PASS" if all(
                row["sample_ids_missing_from_both_metadata_tables"] == 0
                for row in row_counts.values()
            ) else "REVIEW",
            "details": {
                k: v["sample_ids_missing_from_both_metadata_tables"]
                for k, v in row_counts.items()
            },
        },
    }

    for name, result in checks.items():
        graph.execute("""
            INSERT OR REPLACE INTO molecular_import_checks
            (check_name, status, details_json, checked_at_utc)
            VALUES (?, ?, ?, ?)
        """, (name, result["status"], json.dumps(result["details"]), now()))

    graph.commit()

    report = {
        "version": "0.2",
        "created_at_utc": now(),
        "database": str(OUT.relative_to(ROOT)),
        "canonical_sources": row_counts,
        "molecular_observations_by_layer": dict(layer_counts),
        "new_nodes_by_type": dict(feature_counts),
        "total_molecular_observations": observed,
        "total_graph_nodes": graph.execute(
            "SELECT COUNT(*) FROM graph_nodes"
        ).fetchone()[0],
        "total_graph_edges": graph.execute(
            "SELECT COUNT(*) FROM graph_edges"
        ).fetchone()[0],
        "checks": checks,
        "limitations": [
            "OpenPedCan gene symbols and fusion signatures are source labels, not normalized entities.",
            "Sample links to clinical and histology tables use identifier matching only.",
            "Tumour labels are preserved as source fields; no diagnostic crosswalk was applied.",
            "No patient-derived sample was equated with a GDSC cell line.",
            "No drug ranking, efficacy inference, or model training was performed.",
        ],
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    graph.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    graph.close()

    print("\nTYPED EVIDENCE GRAPH v0.2")
    print(f"Molecular observations: {observed:,}")
    print(f"Total nodes: {report['total_graph_nodes']:,}")
    print(f"Total edges: {report['total_graph_edges']:,}")
    print("\nChecks:")
    for name, result in checks.items():
        print(f"  {result['status']:6} {name}: {result['details']}")
    print(f"\nGraph DB: {OUT.relative_to(ROOT)}")
    print(f"Report:   {REPORT.relative_to(ROOT)}")

    if any(item["status"] == "FAIL" for item in checks.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
