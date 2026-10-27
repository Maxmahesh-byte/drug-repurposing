#!/usr/bin/env python3
"""Read-only audit of an existing local research-data repository.

This script does not download, transform, move, or overwrite source data.
It samples tabular files, inventories manifests, and can optionally inspect
trusted locally-created PyTorch graph artifacts. It writes only the report.

Examples (from repository root):
    python scripts/audit_existing_data.py
    python scripts/audit_existing_data.py --sample-rows 2000
    python scripts/audit_existing_data.py --count-rows
    python scripts/audit_existing_data.py --inspect-graphs

NOTE: --inspect-graphs loads .pt files using torch.load(weights_only=False),
which can execute code embedded in a pickle. Use it only for graph artifacts
that you trust and generated locally in this project.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
TABLE_SUFFIXES = {".csv", ".tsv", ".maf", ".txt"}
MANIFEST_MARKER = "manifest"


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def format_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def open_text(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8-sig", errors="replace", newline="")
    return path.open("r", encoding="utf-8-sig", errors="replace", newline="")


def delimiter_for(path: Path) -> str:
    name = path.name.lower()
    if name.endswith((".tsv", ".tsv.gz", ".maf", ".maf.gz")):
        return "\t"
    # CSV is the safest default for other delimited files; avoid sniffer on huge files.
    return ","


def inspect_table(path: Path, sample_rows: int, count_all_rows: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "format": "delimited_text"}
    try:
        delim = delimiter_for(path)
        with open_text(path) as handle:
            reader = csv.reader(handle, delimiter=delim)
            header = next(reader, None)
            if header is None:
                result["error"] = "Empty file"
                return result

            header = [str(cell).strip() for cell in header]
            duplicate_headers = sorted({h for h in header if header.count(h) > 1})
            seen = 0
            empty_counts = [0] * len(header)
            malformed_rows = 0
            total_rows: int | None = 0 if count_all_rows else None
            for row in reader:
                if not row:
                    continue
                seen += 1
                if len(row) != len(header):
                    malformed_rows += 1
                if seen <= sample_rows:
                    for i in range(len(header)):
                        if i >= len(row) or not str(row[i]).strip():
                            empty_counts[i] += 1
                if count_all_rows:
                    assert total_rows is not None
                    total_rows += 1
                elif seen >= sample_rows:
                    # Normal mode samples a bounded number of rows then stops.
                    break

        sampled = min(seen, sample_rows)
        result.update({
            "delimiter": "tab" if delim == "\t" else "comma",
            "columns": header,
            "column_count": len(header),
            "duplicate_columns": duplicate_headers,
            "sample_rows_read": sampled,
            "sample_empty_cell_fraction": round(
                sum(empty_counts) / max(1, sampled * len(header)), 6
            ),
            "sample_empty_cells_by_column": {
                header[i]: empty_counts[i] for i in range(len(header))
                if empty_counts[i] > 0
            },
            "malformed_rows_seen": malformed_rows,
        })
        if count_all_rows:
            result["data_rows"] = total_rows
        else:
            result["data_rows"] = "not counted (use --count-rows to scan whole file)"
    except Exception as exc:
        result["error"] = format_exception(exc)
    return result


def inspect_xlsx(path: Path, sample_rows: int) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "format": "xlsx"}
    try:
        import openpyxl  # optional dependency
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        sheets: list[dict[str, Any]] = []
        for ws in wb.worksheets:
            rows = ws.iter_rows(values_only=True)
            header_values = next(rows, None)
            if header_values is None:
                sheets.append({"name": ws.title, "empty": True})
                continue
            headers = [str(v).strip() if v is not None else f"column_{i+1}"
                       for i, v in enumerate(header_values)]
            inspected = 0
            for _ in rows:
                inspected += 1
                if inspected >= sample_rows:
                    break
            sheets.append({
                "name": ws.title,
                "columns": headers,
                "column_count": len(headers),
                "sample_data_rows_read": inspected,
                "reported_dimensions": ws.calculate_dimension(),
            })
        wb.close()
        result["sheets"] = sheets
    except Exception as exc:
        result["error"] = format_exception(exc)
    return result


def summarize_json(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "format": "json"}
    try:
        with path.open("r", encoding="utf-8") as handle:
            content = json.load(handle)
        result["top_level_type"] = type(content).__name__
        if isinstance(content, dict):
            result["keys"] = sorted(str(k) for k in content.keys())
            # Keep useful scalar metadata without dumping potentially large/nested records.
            safe_scalars: dict[str, Any] = {}
            for key, value in content.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    safe_scalars[str(key)] = value
                elif isinstance(value, list) and len(value) <= 12 and all(
                    isinstance(item, (str, int, float, bool)) or item is None for item in value
                ):
                    safe_scalars[str(key)] = value
            result["scalar_metadata"] = safe_scalars
        elif isinstance(content, list):
            result["items"] = len(content)
    except Exception as exc:
        result["error"] = format_exception(exc)
    return result


def tensor_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(x) for x in shape]
    except Exception:
        return None


def inspect_graph(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "format": "pytorch_artifact"}
    try:
        import torch  # optional dependency
        obj = torch.load(path, map_location="cpu", weights_only=False)
        result["object_type"] = f"{type(obj).__module__}.{type(obj).__name__}"
        if isinstance(obj, dict):
            result["keys"] = sorted(str(k) for k in obj.keys())
            for key, value in obj.items():
                shape = tensor_shape(value)
                if shape is not None:
                    result.setdefault("tensor_shapes", {})[str(key)] = shape
        else:
            for attr in ("num_nodes", "num_edges", "num_node_features"):
                try:
                    val = getattr(obj, attr)
                    if val is not None:
                        result[attr] = int(val)
                except Exception:
                    pass
            keys_attr = getattr(obj, "keys", None)
            if callable(keys_attr):
                try:
                    result["keys"] = sorted(str(k) for k in keys_attr())
                except Exception:
                    pass
            shapes: dict[str, list[int]] = {}
            for attr in ("x", "edge_index", "edge_attr", "y", "node_type", "edge_type"):
                try:
                    shape = tensor_shape(getattr(obj, attr))
                    if shape is not None:
                        shapes[attr] = shape
                except Exception:
                    pass
            if shapes:
                result["tensor_shapes"] = shapes
            # PyG HeteroData has node_types / edge_types.
            for attr in ("node_types", "edge_types"):
                try:
                    val = getattr(obj, attr)
                    if val is not None:
                        result[attr] = [str(x) for x in val]
                except Exception:
                    pass
        del obj
    except Exception as exc:
        result["error"] = format_exception(exc)
    return result


def iter_files(data_root: Path) -> Iterable[Path]:
    for base, dirs, files in os.walk(data_root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for filename in sorted(files):
            path = Path(base) / filename
            if path.is_file():
                yield path


def layer_coverage(paths: list[str]) -> dict[str, bool]:
    roots = {
        "chembl": ("data/raw/chembl/",),
        "experimental_validation": ("data/raw/experimental/",),
        "gdsc": ("data/raw/gdsc/", "data/raw/gdsc_pediatric/", "data/raw/gdsc2_original.csv"),
        "mechanistic": ("data/raw/mechanistic/",),
        "openpedcan": ("data/raw/openpedcan/",),
        "structural": ("data/raw/structural/",),
        "target_evidence": ("data/raw/target/", "data/raw/target_evidence/"),
        "processed_graphs": ("data/processed/",),
    }
    norm = [p.replace("\\", "/") for p in paths]
    return {layer: any(any(p.startswith(root) if root.endswith("/") else p == root
                           for root in roots_for_layer) for p in norm)
            for layer, roots_for_layer in roots.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="repository root (defaults to parent of scripts/)")
    parser.add_argument("--out", type=Path, default=Path("reports/data_audit.json"),
                        help="report path; relative paths are resolved against repository root")
    parser.add_argument("--sample-rows", type=int, default=500,
                        help="max rows sampled per table (default: 500)")
    parser.add_argument("--count-rows", action="store_true",
                        help="scan all delimited rows; may be slow for large files")
    parser.add_argument("--inspect-graphs", action="store_true",
                        help="load .pt graph artifacts (only use trusted local files)")
    parser.add_argument("--no-write", action="store_true", help="print report without writing it")
    args = parser.parse_args()

    root = args.root.resolve()
    data_root = root / "data"
    if not data_root.is_dir():
        print(f"ERROR: data directory not found: {data_root}", file=sys.stderr)
        return 2
    if args.sample_rows < 1:
        parser.error("--sample-rows must be at least 1")

    files = list(iter_files(data_root))
    inventory: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    graphs: list[dict[str, Any]] = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        inventory.append({"path": rel, "bytes": size, "size": human_size(size), "suffix": "".join(path.suffixes)})
        lower = path.name.lower()
        if path.suffix.lower() in TABLE_SUFFIXES or lower.endswith((".csv.gz", ".tsv.gz", ".maf.gz", ".txt.gz")):
            table = inspect_table(path, args.sample_rows, args.count_rows)
            table["path"] = rel
            table["size"] = human_size(size)
            tables.append(table)
        elif path.suffix.lower() == ".xlsx":
            table = inspect_xlsx(path, args.sample_rows)
            table["path"] = rel
            table["size"] = human_size(size)
            tables.append(table)
        if MANIFEST_MARKER in lower and path.suffix.lower() == ".json":
            item = summarize_json(path)
            item["path"] = rel
            manifests.append(item)
        if lower.endswith(".pt") and args.inspect_graphs:
            item = inspect_graph(path)
            item["path"] = rel
            item["size"] = human_size(size)
            graphs.append(item)

    warnings: list[str] = []
    coverage = layer_coverage([x["path"] for x in inventory])
    for layer, exists in coverage.items():
        if not exists:
            warnings.append(f"Expected data layer not detected by path: {layer}")
    for table in tables:
        if table.get("duplicate_columns"):
            warnings.append(f"Duplicate column names in {table['path']}: {table['duplicate_columns']}")
        if table.get("error"):
            warnings.append(f"Could not inspect {table['path']}: {table['error']}")
        if table.get("malformed_rows_seen", 0) > 0:
            warnings.append(f"Uneven field counts seen in sample/scan of {table['path']}: {table['malformed_rows_seen']}")
    for manifest in manifests:
        if manifest.get("error"):
            warnings.append(f"Could not parse manifest {manifest['path']}: {manifest['error']}")
    for graph in graphs:
        if graph.get("error"):
            warnings.append(f"Could not inspect graph {graph['path']}: {graph['error']}")

    total_bytes = sum(x["bytes"] for x in inventory)
    report: dict[str, Any] = {
        "report_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository_root": str(root),
        "data_root": str(data_root),
        "mode": {
            "sample_rows": args.sample_rows,
            "full_row_count_scan": args.count_rows,
            "graph_inspection": args.inspect_graphs,
            "source_data_modified": False,
        },
        "summary": {
            "files": len(inventory),
            "total_size_bytes": total_bytes,
            "total_size_human": human_size(total_bytes),
            "tabular_files_inspected": len(tables),
            "manifests_found": len(manifests),
            "graph_artifacts_inspected": len(graphs),
            "layer_path_coverage": coverage,
            "warnings": len(warnings),
        },
        "inventory": inventory,
        "tables": tables,
        "manifests": manifests,
        "graphs": graphs,
        "warnings": warnings,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f" - {warning}")

    if not args.no_write:
        out_path = args.out if args.out.is_absolute() else root / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nFull report written to: {out_path}")
    else:
        print("\n--no-write set; no report file written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
