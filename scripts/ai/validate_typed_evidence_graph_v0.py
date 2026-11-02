#!/usr/bin/env python3
"""Independent, read-only validation for Typed Evidence Graph v0.1."""

import csv
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path.cwd()
GRAPH = ROOT / "data/ai/typed_evidence_graph_v0.sqlite"
ENGINE = ROOT / "data/ai/evidence_engine_v0.sqlite"
OUT = ROOT / "data/ai/audits/typed_evidence_graph_v0_validation.json"

if not GRAPH.exists() or not ENGINE.exists():
    raise SystemExit("Graph or evidence-engine database is missing.")

g = sqlite3.connect(f"file:{GRAPH}?mode=ro", uri=True)
e = sqlite3.connect(f"file:{ENGINE}?mode=ro", uri=True)
g.row_factory = sqlite3.Row
e.row_factory = sqlite3.Row

checks = []

def check(name, passed, details=None):
    checks.append({
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "details": details if details is not None else {},
    })
    print(f"{'PASS' if passed else 'FAIL'} {name}"
          + (f" — {details}" if details else ""))

def scalar(conn, sql, args=()):
    return conn.execute(sql, args).fetchone()[0]

# 1. Database structure and referential integrity.
integrity = scalar(g, "PRAGMA integrity_check")
check("sqlite_integrity", integrity == "ok", integrity)

fk_errors = g.execute("PRAGMA foreign_key_check").fetchall()
check("foreign_key_integrity", len(fk_errors) == 0,
      f"errors={len(fk_errors)}")

dangling_edges = scalar(g, """
    SELECT COUNT(*)
    FROM graph_edges x
    LEFT JOIN graph_nodes s ON s.node_id = x.source_node_id
    LEFT JOIN graph_nodes t ON t.node_id = x.target_node_id
    WHERE s.node_id IS NULL OR t.node_id IS NULL
""")
check("edge_endpoints_exist", dangling_edges == 0, dangling_edges)

# 2. Every evidence node has exactly one source-membership edge.
bad_source_links = scalar(g, """
    SELECT COUNT(*) FROM graph_nodes n
    WHERE n.node_type = 'EVIDENCE_RECORD'
      AND (
        SELECT COUNT(*) FROM graph_edges x
        WHERE x.target_node_id = n.node_id
          AND x.edge_type = 'SOURCE_HAS_RECORD'
      ) != 1
""")
check("one_source_link_per_evidence_record", bad_source_links == 0,
      f"invalid_records={bad_source_links}")

# Every edge carrying an evidence_record_id must point to a real evidence node.
missing_evidence_nodes = scalar(g, """
    SELECT COUNT(*) FROM graph_edges x
    WHERE x.evidence_record_id IS NOT NULL
      AND NOT EXISTS (
        SELECT 1 FROM graph_nodes n
        WHERE n.node_id = 'EVIDENCE:' || x.evidence_record_id
          AND n.node_type = 'EVIDENCE_RECORD'
      )
""")
check("edge_evidence_references_resolve", missing_evidence_nodes == 0,
      f"unresolved={missing_evidence_nodes}")

# 3. Compare each graph record's saved raw row to the engine's original row.
graph_rows = g.execute("""
    SELECT node_id, attributes_json
    FROM graph_nodes
    WHERE node_type = 'EVIDENCE_RECORD'
    ORDER BY node_id
""")
engine_rows = e.execute("""
    SELECT record_id, raw_row_json
    FROM evidence_records
    ORDER BY record_id
""")

compared = 0
mismatches = 0
missing_graph_records = 0

for graph_row, engine_row in zip(graph_rows, engine_rows):
    compared += 1
    if graph_row["node_id"] != "EVIDENCE:" + engine_row["record_id"]:
        mismatches += 1
        continue
    attributes = json.loads(graph_row["attributes_json"])
    original = json.loads(engine_row["raw_row_json"])
    if attributes.get("raw_source_fields") != original:
        mismatches += 1

engine_count = scalar(e, "SELECT COUNT(*) FROM evidence_records")
graph_count = scalar(g, """
    SELECT COUNT(*) FROM graph_nodes WHERE node_type = 'EVIDENCE_RECORD'
""")
missing_graph_records = max(0, engine_count - compared)

check("graph_records_match_engine",
      engine_count == graph_count
      and compared == engine_count
      and mismatches == 0,
      {
          "engine": engine_count,
          "graph": graph_count,
          "compared": compared,
          "mismatches": mismatches,
          "missing": missing_graph_records,
      })

# 4. Independently recount GDSC2 identifiers and raw disease labels.
gdsc_path = ROOT / "data/raw/gdsc/gdsc2_responses.csv"
drugs, models, cosmic_ids, disease_labels = set(), set(), set(), set()
response_rows = 0
labelled_rows = 0

with gdsc_path.open(newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        response_rows += 1
        for field, target in (
            ("drug_id", drugs),
            ("sanger_model_id", models),
            ("cosmic_id", cosmic_ids),
            ("reported_cancer_type", disease_labels),
        ):
            value = (row.get(field) or "").strip()
            if value:
                target.add(value)
        if (row.get("reported_cancer_type") or "").strip():
            labelled_rows += 1

expected_nodes = {
    "GDSC_DRUG_ID": len(drugs),
    "CELL_LINE_MODEL": len(models),
    "COSMIC_MODEL_ID": len(cosmic_ids),
    "RAW_DISEASE_LABEL": len(disease_labels),
}
node_count_matches = {}
for node_type, expected in expected_nodes.items():
    actual = scalar(g,
        "SELECT COUNT(*) FROM graph_nodes WHERE node_type = ?",
        (node_type,))
    node_count_matches[node_type] = {
        "expected": expected, "actual": actual,
    }

response_edges = scalar(g, """
    SELECT COUNT(*) FROM graph_edges
    WHERE edge_type = 'RECORD_NAMES_GDSC_DRUG'
""")
cell_edges = scalar(g, """
    SELECT COUNT(*) FROM graph_edges
    WHERE edge_type = 'RECORD_TESTED_IN_CELL_MODEL'
""")
disease_edges = scalar(g, """
    SELECT COUNT(*) FROM graph_edges
    WHERE edge_type = 'RECORD_REPORTS_RAW_DISEASE_LABEL'
""")

check("GDSC2_identifier_node_counts_match",
      all(v["expected"] == v["actual"]
          for v in node_count_matches.values()),
      node_count_matches)

check("GDSC2_record_edge_counts_match",
      response_edges == response_rows
      and cell_edges == response_rows
      and disease_edges == labelled_rows,
      {
          "source_rows": response_rows,
          "drug_edges": response_edges,
          "cell_model_edges": cell_edges,
          "source_rows_with_disease_label": labelled_rows,
          "disease_label_edges": disease_edges,
      })

# 5. Quarantined items must connect only to their source file.
review_nodes = scalar(g, """
    SELECT COUNT(*) FROM graph_nodes
    WHERE node_type = 'QUARANTINED_REVIEW_ITEM'
""")
review_queue = scalar(e, "SELECT COUNT(*) FROM review_queue")

review_non_membership_edges = scalar(g, """
    SELECT COUNT(*)
    FROM graph_edges x
    JOIN graph_nodes n
      ON n.node_id = x.source_node_id OR n.node_id = x.target_node_id
    WHERE n.node_type = 'QUARANTINED_REVIEW_ITEM'
      AND x.edge_type != 'SOURCE_HAS_QUARANTINED_REVIEW_ITEM'
""")

check("quarantined_review_items_match",
      review_nodes == review_queue,
      {"graph": review_nodes, "engine": review_queue})

check("quarantine_has_no_entity_edges",
      review_non_membership_edges == 0,
      review_non_membership_edges)

# 6. Ensure no treatment/ranking edges have been introduced.
edge_types = {
    row[0] for row in g.execute(
        "SELECT DISTINCT edge_type FROM graph_edges"
    )
}
prohibited_edge_types = {
    "DRUG_TREATS_DISEASE",
    "DRUG_EFFECTIVE_IN_TUMOUR",
    "RANKED_TREATMENT_FOR",
}
unexpected = sorted(edge_types & prohibited_edge_types)
check("no_drug_effectiveness_or_ranking_edges",
      not unexpected, unexpected)

result = {
    "version": "0.1",
    "checks": checks,
    "gdsc2": {
        "source_rows": response_rows,
        "distinct_drugs": len(drugs),
        "distinct_cell_models": len(models),
        "distinct_cosmic_ids": len(cosmic_ids),
        "distinct_raw_disease_labels": len(disease_labels),
    },
    "graph": {
        "nodes": scalar(g, "SELECT COUNT(*) FROM graph_nodes"),
        "edges": scalar(g, "SELECT COUNT(*) FROM graph_edges"),
    },
    "summary": {
        "passed": sum(c["status"] == "PASS" for c in checks),
        "failed": sum(c["status"] == "FAIL" for c in checks),
    },
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

print(f"\nPassed: {result['summary']['passed']}/{len(checks)}")
print(f"Failed: {result['summary']['failed']}")
print(f"Report: {OUT.relative_to(ROOT)}")

g.close()
e.close()

sys.exit(1 if result["summary"]["failed"] else 0)
