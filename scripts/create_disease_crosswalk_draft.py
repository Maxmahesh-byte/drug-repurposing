import csv
import re
from pathlib import Path

BASE = Path("data/processed/graph_v3")

GDSC_FILE = BASE / "gdsc_disease_vocabulary.csv"
OPENPEDCAN_FILE = BASE / "openpedcan_disease_vocabulary.csv"
OUTPUT = BASE / "gdsc_openpedcan_crosswalk_draft.csv"

# Each component is a candidate concept with possible vocabulary labels.
# These are suggestions for review, NOT approved mappings.
CANDIDATES = {
    "ACC": {
        "kind": "SINGLE_DISEASE_CANDIDATE",
        "components": [
            ["Adrenocortical carcinoma"]
        ],
    },
    "BLCA": {
        "kind": "SINGLE_DISEASE_CANDIDATE",
        "components": [
            ["Bladder Urothelial Carcinoma"]
        ],
    },
    "BRCA": {
        "kind": "BROAD_OR_SUBTYPE_REVIEW",
        "components": [
            ["Breast Invasive Carcinoma"]
        ],
    },
    "CESC": {
        "kind": "COMPOSITE_HISTOLOGY_LABEL",
        "components": [
            [
                "Cervical Squamous Cell Carcinoma and "
                "Endocervical Adenocarcinoma"
            ]
        ],
    },
    "COREAD": {
        "kind": "COMPOSITE_CATEGORY",
        "components": [
            ["Colon Adenocarcinoma"],
            ["Rectum Adenocarcinoma", "Rectal Adenocarcinoma"],
        ],
    },
    "DLBC": {
        "kind": "SINGLE_DISEASE_CANDIDATE",
        "components": [
            [
                "Lymphoid Neoplasm Diffuse Large B-cell Lymphoma",
                "Diffuse Large B-cell Lymphoma",
                "Diffuse Large B-Cell Lymphoma",
            ]
        ],
    },
    "ESCA": {
        "kind": "SINGLE_DISEASE_CANDIDATE",
        "components": [
            ["Esophageal Carcinoma", "Oesophageal Carcinoma"]
        ],
    },
    "GBM": {
        "kind": "NARROWER_SUBTYPE_REVIEW",
        "components": [
            ["Glioblastoma", "Glioblastoma Multiforme"]
        ],
    },
    "HNSC": {
        "kind": "SINGLE_DISEASE_CANDIDATE",
        "components": [
            ["Head and Neck Squamous Cell Carcinoma"]
        ],
    },
    "KIRC": {
        "kind": "SINGLE_DISEASE_CANDIDATE",
        "components": [
            ["Renal Clear Cell Carcinoma",
             "Kidney Renal Clear Cell Carcinoma"]
        ],
    },
    "LCML": {
        "kind": "SINGLE_DISEASE_CANDIDATE",
        "components": [
            ["Chronic Myelogenous Leukemia", "Chronic Myeloid Leukemia"]
        ],
    },
    "LGG": {
        "kind": "BROAD_OR_SUBTYPE_REVIEW",
        "components": [
            ["Low-grade glioma", "Brain Lower Grade Glioma"]
        ],
    },
    "LIHC": {
        "kind": "SINGLE_DISEASE_CANDIDATE",
        "components": [
            ["Hepatocellular Carcinoma",
             "Liver Hepatocellular Carcinoma"]
        ],
    },
    "LUAD": {
        "kind": "SINGLE_DISEASE_CANDIDATE",
        "components": [["Lung Adenocarcinoma"]],
    },
    "LUSC": {
        "kind": "SINGLE_DISEASE_CANDIDATE",
        "components": [["Lung Squamous Cell Carcinoma"]],
    },
    "MESO": {
        "kind": "SINGLE_DISEASE_CANDIDATE",
        "components": [["Mesothelioma"]],
    },
    "MM": {
        "kind": "SINGLE_DISEASE_CANDIDATE",
        "components": [["Multiple Myeloma"]],
    },
    "NB": {
        "kind": "SINGLE_DISEASE_CANDIDATE",
        "components": [["Neuroblastoma"]],
    },
    "OV": {
        "kind": "BROAD_OR_SUBTYPE_REVIEW",
        "components": [["Ovarian Serous Cystadenocarcinoma"]],
    },
    "PAAD": {
        "kind": "SINGLE_DISEASE_CANDIDATE",
        "components": [["Pancreatic Adenocarcinoma"]],
    },
    "PRAD": {
        "kind": "SINGLE_DISEASE_CANDIDATE",
        "components": [["Prostate Adenocarcinoma"]],
    },
    "SARC": {
        "kind": "BROAD_DISEASE_CATEGORY",
        "components": [["Sarcoma"]],
    },
    "SKCM": {
        "kind": "SINGLE_DISEASE_CANDIDATE",
        "components": [
            ["Cutaneous Melanoma", "Skin Cutaneous Melanoma"]
        ],
    },
    "STAD": {
        "kind": "SINGLE_DISEASE_CANDIDATE",
        "components": [["Stomach Adenocarcinoma"]],
    },
    "THCA": {
        "kind": "SINGLE_DISEASE_CANDIDATE",
        "components": [["Thyroid Carcinoma"]],
    },
    "UCEC": {
        "kind": "SINGLE_DISEASE_CANDIDATE",
        "components": [["Uterine Corpus Endometrial Carcinoma"]],
    },
    "ALL": {
        "kind": "SINGLE_DISEASE_CANDIDATE",
        "components": [["Acute Lymphoblastic Leukemia"]],
    },
    "LAML": {
        "kind": "SINGLE_DISEASE_CANDIDATE",
        "components": [["Acute Myeloid Leukemia"]],
    },
    "CLL": {
        "kind": "SINGLE_DISEASE_CANDIDATE",
        "components": [["Chronic Lymphocytic Leukemia"]],
    },
    "MB": {
        "kind": "SINGLE_DISEASE_CANDIDATE",
        "components": [["Medulloblastoma"]],
    },
    "SCLC": {
        "kind": "SINGLE_DISEASE_CANDIDATE",
        "components": [["Small Cell Lung Cancer"]],
    },
}

NO_MAPPING = {"OTHER", "UNCLASSIFIED", "[BLANK]"}


def normalise(value):
    return re.sub(r"[^a-z0-9]", "", value.casefold())


if not GDSC_FILE.is_file() or not OPENPEDCAN_FILE.is_file():
    raise FileNotFoundError("Disease vocabulary files are missing.")

if OUTPUT.exists():
    raise FileExistsError(
        f"Refusing to overwrite existing draft: {OUTPUT}"
    )

with GDSC_FILE.open(encoding="utf-8", newline="") as f:
    gdsc_rows = list(csv.DictReader(f))

with OPENPEDCAN_FILE.open(encoding="utf-8", newline="") as f:
    openpedcan_rows = list(csv.DictReader(f))

openpedcan_labels = {
    normalise(row["openpedcan_tumor_type"]):
    row["openpedcan_tumor_type"]
    for row in openpedcan_rows
}

output_rows = []

for row in gdsc_rows:
    code = row["gdsc_label"].strip()
    spec = CANDIDATES.get(code)

    result = {
        "gdsc_label": code,
        "response_records": row["response_records"],
        "cell_lines": row["cell_lines"],
        "drugs": row["drugs"],
        "candidate_relationship": "",
        "candidate_component_1": "",
        "matched_openpedcan_component_1": "",
        "candidate_component_2": "",
        "matched_openpedcan_component_2": "",
        "mapping_status": "",
        "review_status": "NOT_APPROVED",
        "review_rationale": "",
    }

    if code in NO_MAPPING:
        result["mapping_status"] = "UNMAPPED_NO_DISEASE_CLAIM"
        result["review_rationale"] = (
            "Unclassified, other or blank category; "
            "do not assign a disease label."
        )
        output_rows.append(result)
        continue

    if spec is None:
        result["mapping_status"] = "NO_CANDIDATE_DEFINED"
        result["review_rationale"] = (
            "No candidate definition in this draft; "
            "requires manual review."
        )
        output_rows.append(result)
        continue

    result["candidate_relationship"] = spec["kind"]
    matched_components = 0

    for index, aliases in enumerate(spec["components"], start=1):
        matched = next(
            (
                openpedcan_labels[normalise(alias)]
                for alias in aliases
                if normalise(alias) in openpedcan_labels
            ),
            "",
        )

        result[f"candidate_component_{index}"] = " | ".join(aliases)
        result[f"matched_openpedcan_component_{index}"] = matched

        if matched:
            matched_components += 1

    expected_components = len(spec["components"])

    if matched_components == expected_components:
        result["mapping_status"] = "CANDIDATE_MATCH_FOUND"
    elif matched_components:
        result["mapping_status"] = "PARTIAL_COMPONENT_MATCH"
    else:
        result["mapping_status"] = "NO_EXACT_VOCABULARY_MATCH"

    result["review_rationale"] = (
        "Provisional terminology candidate only. Confirm the "
        "GDSC2 release definition, histological scope, age/model "
        "context and OpenPedCan concept before approving."
    )

    output_rows.append(result)

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(output_rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(output_rows)

print("PROVISIONAL DISEASE CROSSWALK")
print("=" * 55)
print(f"GDSC labels processed: {len(output_rows)}")
print(
    "Candidate matches:",
    sum(r["mapping_status"] == "CANDIDATE_MATCH_FOUND"
        for r in output_rows),
)
print(
    "Partial composite matches:",
    sum(r["mapping_status"] == "PARTIAL_COMPONENT_MATCH"
        for r in output_rows),
)
print(
    "Unmatched / unmapped:",
    sum(r["mapping_status"] in {
        "NO_EXACT_VOCABULARY_MATCH",
        "UNMAPPED_NO_DISEASE_CLAIM",
        "NO_CANDIDATE_DEFINED",
    } for r in output_rows),
)
print(f"\nOutput: {OUTPUT}")
print("Every mapping remains NOT_APPROVED.")
print("No biological or clinical equivalence has been asserted.")
