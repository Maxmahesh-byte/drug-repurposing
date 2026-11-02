#!/usr/bin/env python3
"""Build an evidence-centred, provenance-aware graph from Engine v0.1."""

import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
ENGINE_DB = ROOT / "data/ai/evidence_engine_v0.sqlite"
GRAPH_DB = ROOT / "data/ai/typed_evidence_graph_v0.sqlite"
REPORT = ROOT / "data/ai/audits/typed_evidence_graph_v0_report.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def stable_id(*parts):
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main():
    if not ENGINE_DB.exists():
        raise SystemExit(
            "Evidence Engine database missing. Build and validate v0.1 first."
        )

    GRAPH_DB.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    source = sqlite3.connect(f"file:{ENGINE_DB}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row

    graph = sqlite3.connect(GRAPH_DB)
    graph.execute("PRAGMA foreign_keys = ON")
    graph.execute("PRAGMA journal_mode = WAL")

    # This script rebuilds its own graph tables, not the source evidence tables.
    graph.executescript("""
        DROP TABLE IF EXISTS graph_build_checks;
        DROP TABLE IF EXISTS graph_edges;
        DROP TABLE IF EXISTS graph_nodes;

        CREATE TABLE graph_nodes (
            node_id TEXT PRIMARY KEY,
            node_type TEXT NOT NULL,
            display_label TEXT NOT NULL,
            attributes_json TEXT NOT NULL,
            trust_state TEXT NOT NULL
        );

        CREATE TABLE graph_edges (
            edge_id TEXT PRIMARY KEY,
            source_node_id TEXT NOT NULL REFERENCES graph_nodes(node_id),
            edge_type TEXT NOT NULL,
            target_node_id TEXT NOT NULL REFERENCES graph_nodes(node_id),
            evidence_record_id TEXT,
            assertion_status TEXT NOT NULL,
            provenance_state TEXT NOT NULL,
            source_row_number INTEGER
        );

        CREATE TABLE graph_build_checks (
            check_name TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            details_json TEXT NOT NULL
        );

        CREATE INDEX idx_graph_nodes_type ON graph_nodes(node_type);
        CREATE INDEX idx_graph_edges_type ON graph_edges(edge_type);
        CREATE INDEX idx_graph_edges_source ON graph_edges(source_node_id);
        CREATE INDEX idx_graph_edges_target ON graph_edges(target_node_id);
        CREATE INDEX idx_graph_edges_evidence ON graph_edges(evidence_record_id);
    """)

    node_types = Counter()
    edge_types = Counter()

    def add_node(node_id, node_type, label, attributes, trust_state):
        graph.execute("""
            INSERT OR IGNORE INTO graph_nodes
            (node_id, node_type, display_label, attributes_json, trust_state)
            VALUES (?, ?, ?, ?, ?)
        """, (
            node_id, node_type, str(label),
            json.dumps(attributes, sort_keys=True, ensure_ascii=False),
            trust_state,
        ))
        node_types[node_type] += 1

    def add_edge(source_id, edge_type, target_id, evidence_id,
                 assertion_status, provenance_state, row_number=None):
        edge_id = stable_id(
            source_id, edge_type, target_id, evidence_id, row_number
        )
        graph.execute("""
            INSERT OR IGNORE INTO graph_edges
            (edge_id, source_node_id, edge_type, target_node_id,
             evidence_record_id, assertion_status, provenance_state,
             source_row_number)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            edge_id, source_id, edge_type, target_id,
            evidence_id, assertion_status, provenance_state, row_number,
        ))
        edge_types[edge_type] += 1

    # Register every known source, including sources that have not been
    # admitted into the main evidence graph.
    source_registry = source.execute(
        "SELECT * FROM source_registry ORDER BY path, table_or_sheet"
    ).fetchall()

    source_lookup = {}
    for item in source_registry:
        node_id = f"SOURCE:{item['source_id']}"
        source_lookup[item["source_id"]] = node_id

        add_node(
            node_id,
            "SOURCE_FILE",
            item["path"] + (
                f" [{item['table_or_sheet']}]"
                if item["table_or_sheet"] else ""
            ),
            {
                "source_id": item["source_id"],
                "path": item["path"],
                "table_or_sheet": item["table_or_sheet"],
                "evidence_category": item["evidence_category"],
                "sha256": item["sha256"],
                "hash_state": item["hash_state"],
                "rationale": item["rationale"],
            },
            item["trust_state"],
        )

    # Every ingested observation becomes its own node. Entity relationships
    # point out from that observation, rather than being asserted directly
    # between biological entities.
    evidence_rows = source.execute("""
        SELECT * FROM evidence_records
        ORDER BY source_id, source_row_number, record_id
    """)

    evidence_count = 0
    for record in evidence_rows:
        evidence_count += 1
        raw = json.loads(record["raw_row_json"])
        evidence_id = f"EVIDENCE:{record['record_id']}"
        source_node = source_lookup[record["source_id"]]

        add_node(
            evidence_id,
            "EVIDENCE_RECORD",
            f"{record['evidence_type']} — row {record['source_row_number']}",
            {
                "record_id": record["record_id"],
                "source_id": record["source_id"],
                "source_row_number": record["source_row_number"],
                "native_record_id": record["native_record_id"],
                "evidence_type": record["evidence_type"],
                "permitted_use": record["permitted_use"],
                "raw_source_fields": raw,
            },
            record["provenance_state"],
        )

        add_edge(
            source_node, "SOURCE_HAS_RECORD", evidence_id,
            record["record_id"], "SOURCE_FILE_MEMBERSHIP",
            record["provenance_state"], record["source_row_number"],
        )

        row_no = record["source_row_number"]
        provenance = record["provenance_state"]
        evidence_type = record["evidence_type"]

        if evidence_type == "IN_VITRO_ASSAY_RESPONSE":
            drug_id = (raw.get("drug_id") or "").strip()
            drug_name = (raw.get("drug_name") or "").strip()
            cell_id = (raw.get("sanger_model_id") or "").strip()
            cell_name = (raw.get("cell_line") or "").strip()
            cosmic_id = (raw.get("cosmic_id") or "").strip()
            disease = (raw.get("reported_cancer_type") or "").strip()

            if drug_id:
                node_id = f"DRUG:GDSC:{drug_id}"
                add_node(
                    node_id, "GDSC_DRUG_ID", drug_name or drug_id,
                    {"gdsc_drug_id": drug_id, "source_name": drug_name},
                    "SOURCE_IDENTIFIED",
                )
                add_edge(
                    evidence_id, "RECORD_NAMES_GDSC_DRUG",
                    node_id, record["record_id"], "SOURCE_ASSERTED",
                    provenance, row_no,
                )

            if cell_id:
                node_id = f"MODEL:SANGER:{cell_id}"
                add_node(
                    node_id, "CELL_LINE_MODEL", cell_name or cell_id,
                    {"sanger_model_id": cell_id, "source_name": cell_name},
                    "SOURCE_IDENTIFIED",
                )
                add_edge(
                    evidence_id, "RECORD_TESTED_IN_CELL_MODEL",
                    node_id, record["record_id"], "SOURCE_ASSERTED",
                    provenance, row_no,
                )

            if cosmic_id:
                node_id = f"MODEL:COSMIC:{cosmic_id}"
                add_node(
                    node_id, "COSMIC_MODEL_ID", cosmic_id,
                    {"cosmic_id": cosmic_id},
                    "SOURCE_IDENTIFIED",
                )
                add_edge(
                    evidence_id, "RECORD_NAMES_COSMIC_MODEL_ID",
                    node_id, record["record_id"], "SOURCE_ASSERTED",
                    provenance, row_no,
                )

            if disease:
                # Retain the source vocabulary exactly. Do not automatically
                # equate it to an OpenPedCan diagnosis.
                node_id = f"DISEASE_LABEL:GDSC2:{disease}"
                add_node(
                    node_id, "RAW_DISEASE_LABEL", disease,
                    {
                        "label": disease,
                        "vocabulary_source": "GDSC2",
                        "crosswalk_status": "NOT_RESOLVED_HERE",
                    },
                    "SOURCE_LABEL_UNMAPPED",
                )
                add_edge(
                    evidence_id, "RECORD_REPORTS_RAW_DISEASE_LABEL",
                    node_id, record["record_id"],
                    "SOURCE_ASSERTED_LABEL_NOT_CROSSWALKED",
                    provenance, row_no,
                )

        elif evidence_type == "DRUG_TARGET_ANNOTATION":
            gdsc_id = (raw.get("gdsc_drug_id") or "").strip()
            gdsc_name = (raw.get("gdsc_drug_name") or "").strip()
            molecule_id = (raw.get("molecule_chembl_id") or "").strip()
            target_id = (raw.get("target_chembl_id") or "").strip()
            target_name = (raw.get("target_name") or "").strip()
            target_type = (raw.get("target_type") or "").strip()
            uniprot = (raw.get("uniprot_accession") or "").strip()
            gene = (raw.get("gene_symbol") or "").strip()

            if gdsc_id:
                node_id = f"DRUG:GDSC:{gdsc_id}"
                add_node(
                    node_id, "GDSC_DRUG_ID", gdsc_name or gdsc_id,
                    {"gdsc_drug_id": gdsc_id, "source_name": gdsc_name},
                    "SOURCE_IDENTIFIED",
                )
                add_edge(
                    evidence_id, "ANNOTATION_RECORD_NAMES_GDSC_DRUG",
                    node_id, record["record_id"], "SOURCE_ASSERTED",
                    provenance, row_no,
                )

            if molecule_id:
                node_id = f"MOLECULE:CHEMBL:{molecule_id}"
                add_node(
                    node_id, "CHEMBL_MOLECULE", molecule_id,
                    {"molecule_chembl_id": molecule_id},
                    "SOURCE_IDENTIFIED",
                )
                add_edge(
                    evidence_id, "ANNOTATION_RECORD_NAMES_CHEMBL_MOLECULE",
                    node_id, record["record_id"],
                    "SOURCE_ASSERTED_MAPPING_REQUIRES_REVIEW",
                    provenance, row_no,
                )

            if target_id:
                node_id = f"TARGET:CHEMBL:{target_id}"
                add_node(
                    node_id, "CHEMBL_TARGET", target_name or target_id,
                    {
                        "target_chembl_id": target_id,
                        "target_type": target_type,
                    },
                    "SOURCE_ANNOTATION",
                )
                add_edge(
                    evidence_id, "ANNOTATION_RECORD_NAMES_CHEMBL_TARGET",
                    node_id, record["record_id"],
                    "SOURCE_ASSERTED_TARGET_ANNOTATION",
                    provenance, row_no,
                )

            if uniprot:
                node_id = f"PROTEIN:UNIPROT:{uniprot}"
                add_node(
                    node_id, "UNIPROT_ACCESSION", uniprot,
                    {"uniprot_accession": uniprot},
                    "SOURCE_ANNOTATION",
                )
                add_edge(
                    evidence_id, "ANNOTATION_RECORD_NAMES_UNIPROT",
                    node_id, record["record_id"], "SOURCE_ASSERTED",
                    provenance, row_no,
                )

            if gene:
                node_id = f"GENE_SYMBOL:AS_RECORDED:{gene}"
                add_node(
                    node_id, "GENE_SYMBOL_AS_RECORDED", gene,
                    {"gene_symbol": gene, "normalization": "NOT_PERFORMED"},
                    "SOURCE_ANNOTATION",
                )
                add_edge(
                    evidence_id, "ANNOTATION_RECORD_NAMES_GENE_SYMBOL",
                    node_id, record["record_id"], "SOURCE_ASSERTED",
                    provenance, row_no,
                )

    # Quarantined rows appear only as review items. They are not connected
    # to drug, target, gene, model, or disease entities.
    review_rows = source.execute("""
        SELECT * FROM review_queue ORDER BY source_id, source_row_number
    """)

    review_count = 0
    for item in review_rows:
        review_count += 1
        review_id = f"REVIEW:{item['review_id']}"
        source_node = source_lookup[item["source_id"]]

        add_node(
            review_id,
            "QUARANTINED_REVIEW_ITEM",
            f"{item['source_row_number']} — pending verification",
            {
                "review_id": item["review_id"],
                "source_id": item["source_id"],
                "source_row_number": item["source_row_number"],
                "review_reason": item["review_reason"],
                "raw_source_fields": json.loads(item["raw_row_json"]),
            },
            "QUARANTINED_PENDING_REVIEW",
        )
        add_edge(
            source_node, "SOURCE_HAS_QUARANTINED_REVIEW_ITEM",
            review_id, None, "REVIEW_QUEUE_MEMBERSHIP",
            "QUARANTINED_PENDING_REVIEW", item["source_row_number"],
        )

    # Structural checks.
    dangling = graph.execute("""
        SELECT COUNT(*) FROM graph_edges e
        LEFT JOIN graph_nodes s ON e.source_node_id = s.node_id
        LEFT JOIN graph_nodes t ON e.target_node_id = t.node_id
        WHERE s.node_id IS NULL OR t.node_id IS NULL
    """).fetchone()[0]

    evidence_nodes = graph.execute("""
        SELECT COUNT(*) FROM graph_nodes WHERE node_type = 'EVIDENCE_RECORD'
    """).fetchone()[0]

    review_nodes = graph.execute("""
        SELECT COUNT(*) FROM graph_nodes
        WHERE node_type = 'QUARANTINED_REVIEW_ITEM'
    """).fetchone()[0]

    unmapped_disease_nodes = graph.execute("""
        SELECT COUNT(*) FROM graph_nodes
        WHERE node_type = 'RAW_DISEASE_LABEL'
    """).fetchone()[0]

    quarantine_entity_edges = graph.execute("""
        SELECT COUNT(*) FROM graph_edges e
        JOIN graph_nodes n ON e.source_node_id = n.node_id
        WHERE n.node_type = 'QUARANTINED_REVIEW_ITEM'
          AND e.edge_type NOT LIKE 'SOURCE_HAS_QUARANTINED_%'
    """).fetchone()[0]

    checks = {
        "no_dangling_edges": {
            "status": "PASS" if dangling == 0 else "FAIL",
            "count": dangling,
        },
        "evidence_record_nodes_match_engine": {
            "status": "PASS" if evidence_nodes == evidence_count else "FAIL",
            "graph": evidence_nodes,
            "engine": evidence_count,
        },
        "review_items_match_engine": {
            "status": "PASS" if review_nodes == review_count else "FAIL",
            "graph": review_nodes,
            "engine": review_count,
        },
        "quarantine_entities_not_linked": {
            "status": "PASS" if quarantine_entity_edges == 0 else "FAIL",
            "count": quarantine_entity_edges,
        },
        "disease_labels_preserved_unmapped": {
            "status": "INFO",
            "count": unmapped_disease_nodes,
        },
        "graph_is_descriptive_not_predictive": {
            "status": "PASS",
            "note": "No model training, inferred biological edges, or drug ranking performed.",
        },
    }

    for name, detail in checks.items():
        graph.execute("""
            INSERT OR REPLACE INTO graph_build_checks
            (check_name, status, details_json) VALUES (?, ?, ?)
        """, (name, detail["status"], json.dumps(detail, sort_keys=True)))

    graph.commit()

    node_counts = {
        row[0]: row[1]
        for row in graph.execute("""
            SELECT node_type, COUNT(*) FROM graph_nodes GROUP BY node_type
        """)
    }
    edge_counts = {
        row[0]: row[1]
        for row in graph.execute("""
            SELECT edge_type, COUNT(*) FROM graph_edges GROUP BY edge_type
        """)
    }

    report = {
        "version": "0.1",
        "created_at_utc": utc_now(),
        "database": str(GRAPH_DB.relative_to(ROOT)),
        "nodes_by_type": node_counts,
        "edges_by_type": edge_counts,
        "total_nodes": sum(node_counts.values()),
        "total_edges": sum(edge_counts.values()),
        "checks": checks,
        "scope": [
            "GDSC2 source response records",
            "ChEMBL drug-target annotation records",
            "all registered source files as catalog nodes",
            "quarantined rows as isolated review items",
        ],
        "not_yet_integrated": [
            "OpenPedCan sample-level mutations, copy-number changes and fusions",
            "approved sample/cell-line/disease crosswalks",
            "verified structure and pathway evidence",
            "literature evidence",
        ],
        "limitations": [
            "This graph preserves source assertions; it does not verify all biology.",
            "GDSC2 disease labels remain in their original vocabulary.",
            "GDSC and ChEMBL drug identifiers remain separate node types.",
            "No inferred cross-source equivalence edges are added.",
            "No drug efficacy prediction, recommendation, or ranking is performed.",
        ],
    }

    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("TYPED EVIDENCE GRAPH v0.1")
    print(f"Nodes: {report['total_nodes']:,}")
    print(f"Edges: {report['total_edges']:,}")
    print("\nNode types:")
    for name, count in sorted(node_counts.items()):
        print(f"  {name}: {count:,}")
    print("\nEdge types:")
    for name, count in sorted(edge_counts.items()):
        print(f"  {name}: {count:,}")
    print("\nChecks:")
    for name, detail in checks.items():
        print(f"  {detail['status']:6} {name}: {detail}")
    print(f"\nGraph DB: {GRAPH_DB.relative_to(ROOT)}")
    print(f"Report:   {REPORT.relative_to(ROOT)}")

    graph.close()
    source.close()

    if any(d["status"] == "FAIL" for d in checks.values()):
        raise SystemExit("Graph structural validation failed.")


if __name__ == "__main__":
    main()
