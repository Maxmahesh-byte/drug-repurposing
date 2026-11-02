import csv
from pathlib import Path

ROOT = Path.cwd()
OUT = ROOT / "data/ai/audits/evidence_provenance_gate.csv"

def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def norm(value):
    return (value or "").strip().casefold()

def exact_values(path, column):
    return {
        norm(r.get(column))
        for r in read_csv(path)
        if norm(r.get(column))
    }

openpedcan_path = ROOT / "data/raw/openpedcan/openpedcan_phase1.csv"
gdsc_path = ROOT / "data/raw/gdsc/gdsc2_responses.csv"

if not openpedcan_path.exists() or not gdsc_path.exists():
    raise SystemExit("Required source file missing; run from the project root.")

pediatric_samples = exact_values(openpedcan_path, "sample_id")
gdsc = read_csv(gdsc_path)
gdsc_lines = {norm(r.get("cell_line")) for r in gdsc if norm(r.get("cell_line"))}
gdsc_drugs = {norm(r.get("drug_name")) for r in gdsc if norm(r.get("drug_name"))}

targets = [
    ("data/raw/experimental/experimental_validation_phase7.csv", "experimental_claim"),
    ("data/raw/mechanistic/chembl_reactome_string_phase4.csv", "mechanistic_annotation"),
    ("data/raw/structural/pdb_structural_phase6.csv", "structural_annotation"),
    ("data/raw/target/target_phase2.csv", "sample_molecular_annotation"),
    ("data/raw/target_evidence/bindingdb_go_opentargets_phase5.csv", "target_annotation"),
    ("data/raw/gdsc_pediatric/gdsc_pediatric_phase3.csv", "response_annotation"),
    ("data/raw/new_batch_2026.csv", "response_annotation"),
]

# Public structure records already checked for this review.
# These flags establish a discrepancy, not a biological conclusion.
pdb_flags = {
    "4O2B": "REVIEW_REQUIRED: registered as tubulin-colchicine complex",
    "1Z3R": "REVIEW_REQUIRED: registered as viral envelope-protein domain",
    "4BKX": "REVIEW_REQUIRED: HDAC1-MTA1 structure; claimed drug binding not established",
}

report = []

for relative, category in targets:
    path = ROOT / relative
    if not path.exists():
        report.append({
            "file": relative, "row": "", "category": category,
            "sample_id": "", "cell_line": "", "drug_name": "",
            "pdb_id": "", "identifier_check": "FILE_MISSING",
            "provenance_check": "NOT_VERIFIED",
            "review_note": "Source file is absent",
        })
        continue

    for i, row in enumerate(read_csv(path), start=2):
        sample = row.get("sample_id", "")
        line = row.get("cell_line", row.get("CELL_LINE", ""))
        drug = row.get("drug_name", row.get("DRUG_NAME", ""))
        pdb = (row.get("pdb_id") or "").strip().upper()

        checks = []
        notes = []

        if sample:
            if norm(sample) in pediatric_samples:
                checks.append("sample_id_exact_match_in_openpedcan")
            else:
                checks.append("sample_id_not_found_by_exact_match_in_openpedcan")
                notes.append("Sample identity needs review; do not infer equivalence.")

        if line:
            if norm(line) in gdsc_lines:
                checks.append("cell_line_exact_match_in_gdsc")
            else:
                checks.append("cell_line_not_found_by_exact_name_in_gdsc")
                notes.append("Could be an alias or absent; exact-name failure is not proof.")

        if drug:
            if norm(drug) in gdsc_drugs:
                checks.append("drug_name_exact_match_in_gdsc")
            else:
                checks.append("drug_name_not_found_by_exact_name_in_gdsc")
                notes.append("Name reconciliation is required; do not auto-map.")

        if pdb:
            if pdb in pdb_flags:
                checks.append("pdb_record_requires_review")
                notes.append(pdb_flags[pdb])
            else:
                checks.append("pdb_mapping_not_checked_in_this_audit")

        # A generic source label or score does not establish provenance.
        source_fields = [
            k for k in row
            if k.lower() in {"source_url", "doi", "pmid", "accession",
                             "record_id", "study_id", "assay_id"}
        ]
        populated_source_fields = [
            k for k in source_fields if (row.get(k) or "").strip()
        ]

        if populated_source_fields:
            provenance = "SOURCE_IDENTIFIER_PRESENT_BUT_STILL_REQUIRES_VERIFICATION"
        else:
            provenance = "NO_DIRECT_SOURCE_IDENTIFIER_FOUND"

        if any("score" in k.lower() or "confidence" in k.lower()
               for k in row):
            notes.append(
                "Score/confidence value is not assumed calibrated or experimentally measured."
            )

        if category in {"experimental_claim", "mechanistic_annotation",
                        "structural_annotation", "target_annotation"}:
            notes.append(
                "Keep excluded from validated evidence until the original record and claim are verified."
            )

        report.append({
            "file": relative,
            "row": i,
            "category": category,
            "sample_id": sample,
            "cell_line": line,
            "drug_name": drug,
            "pdb_id": pdb,
            "identifier_check": "; ".join(checks) or "NO_CROSSWALKABLE_ID_FOUND",
            "provenance_check": provenance,
            "review_note": " ".join(notes),
        })

OUT.parent.mkdir(parents=True, exist_ok=True)
fields = [
    "file", "row", "category", "sample_id", "cell_line", "drug_name",
    "pdb_id", "identifier_check", "provenance_check", "review_note",
]
with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(report)

print(f"Rows reviewed: {len(report)}")
print(f"Report: {OUT.relative_to(ROOT)}")
print("\nImportant: exact-name matches are not proof of biological equivalence.")
print("Treat flagged rows as unverified until a reviewer checks the source.")
