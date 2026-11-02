#!/usr/bin/env python3

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/ai/baselines"
REGISTRY = ROOT / "data/ai/experiments/registry.jsonl"
RUNS = ROOT / "data/ai/experiments/runs"

VALIDATION = BASE / "gdsc2_baseline_release_validation.json"
MANIFEST = BASE / "gdsc2_baseline_release_manifest.json"
LN_METRICS = BASE / "gdsc2_ln_ic50_baseline_metrics.json"
AUC_METRICS = BASE / "gdsc2_auc_baseline_metrics.json"
LN_DIAGNOSTICS = BASE / "gdsc2_baseline_diagnostic_report.json"

def read_json(path):
    if not path.exists():
        raise SystemExit(f"Required file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    validation = read_json(VALIDATION)
    manifest = read_json(MANIFEST)
    ln = read_json(LN_METRICS)
    auc = read_json(AUC_METRICS)
    diagnostics = read_json(LN_DIAGNOSTICS)

    if validation.get("status") != "PASS":
        raise SystemExit("Refusing to register: validation is not PASS.")

    if manifest.get("status") != "PASS":
        raise SystemExit("Refusing to register: release manifest is not PASS.")

    if ln.get("target") != "ln_ic50":
        raise SystemExit("Unexpected LN(IC50) metrics target.")

    if auc.get("target") != "auc":
        raise SystemExit("Unexpected AUC metrics target.")

    if ln.get("rows") != auc.get("rows"):
        raise SystemExit("LN(IC50) and AUC row counts differ.")

    ln_split = ln.get("split", {})
    auc_split = auc.get("split", {})

    if ln_split.get("n_splits") != 5:
        raise SystemExit("Expected five LN(IC50) folds.")

    if auc.get("split_method") != "GroupKFold by Sanger model ID":
        raise SystemExit("Unexpected AUC split definition.")

    if not validation.get("checks"):
        raise SystemExit("Validation report contains no checks.")

    failed = [
        check for check in validation["checks"]
        if check.get("passed") is not True
    ]
    if failed:
        raise SystemExit(f"Validation checks failed: {failed}")

    timestamp = datetime.now(timezone.utc)
    timestamp_text = timestamp.isoformat()

    manifest_hash = sha256(MANIFEST)
    run_id = (
        "GDSC2_BASELINE_"
        + timestamp.strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + manifest_hash[:8]
    )

    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    record = {
        "run_id": run_id,
        "created_utc": timestamp_text,
        "experiment_type": "descriptive_response_baseline",
        "status": "REGISTERED",
        "source_release": "GDSC2 2023-10-27",
        "source_rows": manifest.get("source_rows"),
        "source_manifest_sha256": manifest_hash,
        "validation_status": validation["status"],
        "validation_checks": [
            check["name"] for check in validation["checks"]
        ],
        "split": {
            "method": "GroupKFold",
            "group": "sanger_model_id",
            "folds": 5,
            "same_fold_assignments_for_endpoints": True,
        },
        "targets": {
            "ln_ic50": {
                "metrics": ln["overall_metrics"],
                "diagnostic_macro_average_drug_mae":
                    diagnostics.get("macro_average_drug_mae"),
            },
            "auc": {
                "metrics": auc["overall_metrics"],
                "macro_average_drug_mae":
                    auc.get("macro_average_drug_mae"),
            },
        },
        "baseline_comparison": {
            "ln_ic50_B2_vs_B1":
                diagnostics.get("B2_vs_B1"),
            "auc_B2_vs_B1":
                auc.get("B2_vs_B1"),
        },
        "model_definitions": manifest.get("baseline_models", {}),
        "leakage_controls": {
            "target_used_as_input": False,
            "cell_line_grouped_split": True,
            "held_out_predictions_recomputed": True,
            "metrics_recomputed": True,
        },
        "limitations": [
            "Baseline uses drug identity and reported cancer type, "
            "not molecular features.",
            "Evaluation is for held-out cell lines with drugs represented "
            "in training.",
            "Unseen-drug generalisation has not been tested.",
            "This is experimental response estimation, not clinical efficacy.",
        ],
    }

    # Store a run-specific record.
    record_path = run_dir / "experiment_record.json"
    record_path.write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )

    # Hash the record and important reports for traceability.
    artifacts = [
        VALIDATION,
        MANIFEST,
        LN_METRICS,
        AUC_METRICS,
        LN_DIAGNOSTICS,
        BASE / "gdsc2_ln_ic50_oof_predictions.csv",
        BASE / "gdsc2_auc_oof_predictions.csv",
        BASE / "gdsc2_ln_ic50_per_drug.csv",
        BASE / "gdsc2_auc_per_drug.csv",
    ]

    missing = [str(p) for p in artifacts if not p.exists()]
    if missing:
        raise SystemExit(f"Missing expected artifacts: {missing}")

    artifact_hashes = {
        str(path.relative_to(ROOT)): {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in artifacts
    }

    record["artifacts"] = artifact_hashes
    record_path.write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )

    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "run_id": run_id,
            "created_utc": timestamp_text,
            "status": "REGISTERED",
            "record_path": str(record_path.relative_to(ROOT)),
            "source_manifest_sha256": manifest_hash,
        }) + "\n")

    print("\n========== EXPERIMENT REGISTERED ==========")
    print(f"Run ID: {run_id}")
    print(f"Validation: {validation['status']}")
    print(f"LN(IC50) B2 MAE: "
          f"{ln['overall_metrics']['B2_drug_cancer_median']['mae']}")
    print(f"AUC B2 MAE: "
          f"{auc['overall_metrics']['B2_drug_cancer_median']['mae']}")
    print(f"Artifacts hashed: {len(artifact_hashes)}")
    print(f"Run record: {record_path}")
    print(f"Registry: {REGISTRY}")
    print("\nExisting benchmark predictions unchanged.")

if __name__ == "__main__":
    main()
