#!/usr/bin/env python3
"""Inventory research-data schemas without joining datasets or training models."""

from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
SEARCH_ROOTS = [ROOT / "data" / "raw", ROOT / "data" / "processed"]
OUT_DIR = ROOT / "data" / "ai" / "audits"
OUT_CSV = OUT_DIR / "molecular_feature_inventory.csv"
OUT_JSON = OUT_DIR / "molecular_feature_inventory_summary.json"

MAX_COUNT_BYTES = 128 * 1024 * 1024
SUPPORTED_SUFFIXES = (
    ".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz",
    ".parquet", ".feather", ".xlsx", ".xls", ".sqlite", ".db",
)

ID_HINTS = re.compile(
    r"(^id$|_id$|sample|specimen|biospecimen|patient|case|subject|"
    r"cell.?line|sanger|cosmic|drug|molecule|chembl|target|uniprot|"
    r"gene|hugo|hgnc|entrez|ensembl|variant|fusion|source|accession)",
    re.I,
)
OUTCOME_HINTS = re.compile(
    r"(^|_)(ln_ic50|ic50|auc|response|response_value|sensitivity|"
    r"resistance|viability|effect|outcome|score)(_|$)",
    re.I,
)


def classify(path_text: str, columns: list[str]) -> str:
    text = (path_text + " " + " ".join(columns)).lower()

    if any(x in text for x in ("crosswalk", "mapping_candidate", "mapping_review")):
        return "identity_crosswalk"
    if any(x in text for x in ("gdsc", "ln_ic50", "ic50", "dose_response")):
        if any(x in text for x in ("target", "chembl", "mechanism", "uniprot")):
            return "drug_target_annotation"
        return "experimental_drug_response"
    if any(x in text for x in ("mutation", "variant", "maf", "intergenic")):
        return "sequence_variant"
    if any(x in text for x in ("copy_number", "cnv", "amplification", "deletion")):
        return "copy_number"
    if any(x in text for x in ("fusion", "rearrangement")):
        return "gene_fusion"
    if any(x in text for x in ("expression", "transcript", "rna_seq", "gene_abundance")):
        return "gene_expression"
    if any(x in text for x in ("chembl", "drug_target", "target_evidence", "mechanism")):
        return "drug_target_annotation"
    if any(x in text for x in ("clinical", "histolog", "diagnos", "phenotyp", "openpedcan")):
        return "clinical_or_disease_annotation"
    if any(x in text for x in ("molecular", "protein", "pathway", "gene")):
        return "molecular_annotation"
    return "other_tabular_or_database"


def supported_file(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)


def delimiter_for(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".tsv") or name.endswith(".tsv.gz"):
        return "\t"
    return ","


def delimited_schema(path: Path) -> tuple[list[str], int | None, str]:
    import gzip

    opener = gzip.open if path.name.lower().endswith(".gz") else open
    kwargs = {"mode": "rt", "encoding": "utf-8-sig", "errors": "replace", "newline": ""}
    with opener(path, **kwargs) as handle:
        reader = csv.reader(handle, delimiter=delimiter_for(path))
        header = next(reader, [])
        if not header:
            return [], None, "empty_or_no_header"

        # Count records only for reasonably small, uncompressed files.
        row_count = None
        if path.stat().st_size <= MAX_COUNT_BYTES and not path.name.lower().endswith(".gz"):
            row_count = 0
            for _ in reader:
                row_count += 1

    return [str(c).strip() for c in header], row_count, "ok"


def sqlite_schemas(path: Path) -> list[dict]:
    result = []
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        tables = connection.execute("""
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """).fetchall()

        for (table_name,) in tables:
            escaped = table_name.replace('"', '""')
            columns = [
                row[1]
                for row in connection.execute(f'PRAGMA table_info("{escaped}")')
            ]
            result.append({"table": table_name, "columns": columns})
    finally:
        connection.close()
    return result


def inspect_file(path: Path) -> list[dict]:
    relative = str(path.relative_to(ROOT))
    stat = path.stat()
    base = {
        "path": relative,
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
    }
    name = path.name.lower()

    try:
        if name.endswith((".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz")):
            columns, row_count, status = delimited_schema(path)
            return [{
                **base,
                "table_or_sheet": "",
                "format": "delimited_text",
                "row_count": row_count if row_count is not None else "",
                "column_count": len(columns),
                "columns": " | ".join(columns),
                "candidate_id_columns": " | ".join(
                    c for c in columns if ID_HINTS.search(c)
                ),
                "outcome_like_columns": " | ".join(
                    c for c in columns if OUTCOME_HINTS.search(c.lower())
                ),
                "evidence_category": classify(relative, columns),
                "status": status if row_count is not None else f"{status};large_file_row_count_not_scanned",
            }]

        if name.endswith((".sqlite", ".db")):
            schemas = sqlite_schemas(path)
            rows = []
            for item in schemas:
                columns = item["columns"]
                rows.append({
                    **base,
                    "table_or_sheet": item["table"],
                    "format": "sqlite_table",
                    "row_count": "",
                    "column_count": len(columns),
                    "columns": " | ".join(columns),
                    "candidate_id_columns": " | ".join(
                        c for c in columns if ID_HINTS.search(c)
                    ),
                    "outcome_like_columns": " | ".join(
                        c for c in columns if OUTCOME_HINTS.search(c.lower())
                    ),
                    "evidence_category": classify(
                        relative + " " + item["table"], columns
                    ),
                    "status": "schema_only;no_table_rows_read",
                })
            if rows:
                return rows
            return [{
                **base, "table_or_sheet": "", "format": "sqlite",
                "row_count": "", "column_count": 0, "columns": "",
                "candidate_id_columns": "", "outcome_like_columns": "",
                "evidence_category": classify(relative, []),
                "status": "no_user_tables_found",
            }]

        if name.endswith((".xlsx", ".xls")):
            try:
                from openpyxl import load_workbook
                workbook = load_workbook(path, read_only=True, data_only=True)
                rows = []
                for sheet in workbook.worksheets:
                    first_row = next(
                        sheet.iter_rows(min_row=1, max_row=1, values_only=True),
                        (),
                    )
                    columns = [
                        str(value).strip() if value is not None else ""
                        for value in first_row
                    ]
                    while columns and not columns[-1]:
                        columns.pop()
                    rows.append({
                        **base,
                        "table_or_sheet": sheet.title,
                        "format": "excel_sheet",
                        "row_count": "",
                        "column_count": len(columns),
                        "columns": " | ".join(columns),
                        "candidate_id_columns": " | ".join(
                            c for c in columns if ID_HINTS.search(c)
                        ),
                        "outcome_like_columns": " | ".join(
                            c for c in columns if OUTCOME_HINTS.search(c.lower())
                        ),
                        "evidence_category": classify(relative, columns),
                        "status": "header_only;row_count_not_scanned",
                    })
                workbook.close()
                return rows
            except Exception as exc:
                return [{
                    **base, "table_or_sheet": "", "format": "excel",
                    "row_count": "", "column_count": 0, "columns": "",
                    "candidate_id_columns": "", "outcome_like_columns": "",
                    "evidence_category": classify(relative, []),
                    "status": f"excel_schema_error:{type(exc).__name__}:{exc}",
                }]

        # Parquet/Feather support is optional.
        try:
            import pyarrow.parquet as pq
            if name.endswith(".parquet"):
                metadata = pq.ParquetFile(path).schema.names
                row_count = pq.ParquetFile(path).metadata.num_rows
                columns = list(metadata)
                return [{
                    **base, "table_or_sheet": "", "format": "parquet",
                    "row_count": row_count, "column_count": len(columns),
                    "columns": " | ".join(columns),
                    "candidate_id_columns": " | ".join(
                        c for c in columns if ID_HINTS.search(c)
                    ),
                    "outcome_like_columns": " | ".join(
                        c for c in columns if OUTCOME_HINTS.search(c.lower())
                    ),
                    "evidence_category": classify(relative, columns),
                    "status": "metadata_only",
                }]
        except ImportError:
            pass

        return [{
            **base, "table_or_sheet": "", "format": path.suffix,
            "row_count": "", "column_count": 0, "columns": "",
            "candidate_id_columns": "", "outcome_like_columns": "",
            "evidence_category": classify(relative, []),
            "status": "format_not_inspected",
        }]

    except Exception as exc:
        return [{
            **base, "table_or_sheet": "", "format": path.suffix,
            "row_count": "", "column_count": 0, "columns": "",
            "candidate_id_columns": "", "outcome_like_columns": "",
            "evidence_category": classify(relative, []),
            "status": f"inspection_error:{type(exc).__name__}:{exc}",
        }]


def main() -> None:
    if not (ROOT / "data").exists():
        raise SystemExit(
            "Run this script from the project root containing the data/ directory."
        )

    files = []
    for search_root in SEARCH_ROOTS:
        if not search_root.exists():
            continue
        for directory, subdirs, filenames in os.walk(search_root):
            subdirs[:] = [
                d for d in subdirs
                if d not in {".git", "__pycache__", "node_modules"}
            ]
            for filename in filenames:
                path = Path(directory) / filename
                if supported_file(path):
                    files.append(path)

    inventory = []
    for path in sorted(files):
        inventory.extend(inspect_file(path))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path", "table_or_sheet", "format", "size_bytes", "modified_utc",
        "row_count", "column_count", "columns", "candidate_id_columns",
        "outcome_like_columns", "evidence_category", "status",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inventory)

    categories = {}
    statuses = {}
    for record in inventory:
        category = record["evidence_category"]
        statuses[record["status"]] = statuses.get(record["status"], 0) + 1
        categories[category] = categories.get(category, 0) + 1

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "search_roots": [str(p) for p in SEARCH_ROOTS if p.exists()],
        "files_discovered": len(files),
        "inventory_rows": len(inventory),
        "rows_by_evidence_category": categories,
        "rows_by_status": statuses,
        "outputs": [str(OUT_CSV.relative_to(ROOT)), str(OUT_JSON.relative_to(ROOT))],
        "limitations": [
            "Schema inventory is not biological or clinical validation.",
            "Large delimited files are not fully scanned just to count rows.",
            "Candidate identifier columns are heuristics and require review.",
            "An absent annotation or target row is not proof that the biology is absent.",
            "No datasets were joined and no model was trained."
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("MOLECULAR FEATURE INVENTORY")
    print(f"Files discovered: {len(files):,}")
    print(f"Inventory rows:   {len(inventory):,}")
    print("\nEvidence categories:")
    for category, count in sorted(categories.items()):
        print(f"  {category}: {count:,}")
    print("\nStatus counts:")
    for status, count in sorted(statuses.items()):
        print(f"  {status}: {count:,}")
    print(f"\nCSV:  {OUT_CSV.relative_to(ROOT)}")
    print(f"JSON: {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
