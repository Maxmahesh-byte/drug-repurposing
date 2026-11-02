import csv
import gzip
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

report = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "repository": str(ROOT),
    "data_directory_exists": DATA.exists(),
    "total_files": 0,
    "total_bytes": 0,
    "directories": {},
    "files": [],
    "errors": [],
}

def inspect_file(path):
    relative = str(path.relative_to(ROOT))
    size = path.stat().st_size

    entry = {
        "path": relative,
        "size_bytes": size,
        "extension": "".join(path.suffixes[-2:]),
    }

    report["total_files"] += 1
    report["total_bytes"] += size

    try:
        if path.name.endswith(".json"):
            with path.open(encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                entry["json_keys"] = list(obj.keys())

        elif path.suffix.lower() == ".csv":
            with path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as f:
                reader = csv.reader(f)
                entry["columns"] = next(reader, [])

        elif path.suffix.lower() == ".tsv":
            with path.open("r", encoding="utf-8-sig") as f:
                entry["columns"] = f.readline().rstrip("\n").split("\t")

        elif path.suffix.lower() == ".gz":
            with gzip.open(path, "rt", encoding="utf-8-sig") as f:
                entry["columns"] = f.readline().rstrip("\n").split("\t")

        elif path.suffix.lower() in (".pt", ".pth"):
            entry["note"] = "Serialized graph/model artifact; not loaded."

        elif path.suffix.lower() in (".xlsx", ".xls"):
            entry["note"] = "Spreadsheet; not opened during this audit."

    except Exception as exc:
        entry["inspection_error"] = str(exc)
        report["errors"].append({
            "path": relative,
            "error": str(exc)
        })

    report["files"].append(entry)


if not DATA.exists():
    raise SystemExit("ERROR: data/ directory not found.")

for path in sorted(DATA.rglob("*")):
    if path.is_file():
        inspect_file(path)

for directory in sorted(p for p in DATA.iterdir() if p.is_dir()):
    files = [p for p in directory.rglob("*") if p.is_file()]
    report["directories"][directory.name] = {
        "file_count": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
    }

REPORTS.mkdir(exist_ok=True)
output = REPORTS / "data_audit.json"
output.write_text(json.dumps(report, indent=2))

print("\n========== DATA AUDIT ==========")
print("Files:", report["total_files"])
print("Total size (GB):", round(report["total_bytes"] / 1e9, 3))
print("Errors:", len(report["errors"]))

print("\nDATA DIRECTORIES:")
for name, info in report["directories"].items():
    print(
        f"{name}: {info['file_count']} files, "
        f"{info['total_bytes'] / 1e6:.1f} MB"
    )

print("\nGRAPH ARTIFACTS:")
for entry in report["files"]:
    if entry["path"].endswith((".pt", ".pth")):
        print(
            f"{entry['path']} "
            f"({entry['size_bytes'] / 1e6:.1f} MB)"
        )

print("\nAudit report:", output)
print("================================")
