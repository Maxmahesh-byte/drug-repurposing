#!/usr/bin/env python3
"""Run additional fixed-fold GDSC2 ablation seeds and analyze them together.

By default, seed 42 is reused from the already-completed reference run and
seeds 43-46 are trained sequentially. Every new seed gets a separate output
directory. The existing OOF predictions are rechecked for identical curve IDs,
outer folds, and observed targets before the uncertainty report is produced.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/ai/run_gdsc2_pair_gnn_ablations.py"
ANALYZER = ROOT / "scripts/ai/analyze_gdsc2_pair_gnn_uncertainty.py"
DEFAULT_REFERENCE = (
    ROOT / "data/ai/gdsc2_pair_gnn_ablations/"
    "gdsc2_pair_gnn_ablation_oof_predictions.csv"
)


def validate_oof_file(path: Path, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"OOF prediction file not found: {path}")
    frame = pd.read_csv(
        path,
        dtype={
            "curve_id": str,
            "drug_id": str,
            "sanger_model_id": str,
        },
    )
    required = {
        "curve_id", "sanger_model_id", "fold", "actual_ln_ic50", "actual_auc",
        "pred_full_graph_ln_ic50", "pred_full_graph_auc",
        "pred_drug_id_only_ln_ic50", "pred_drug_id_only_auc",
        "pred_expression_only_ln_ic50", "pred_expression_only_auc",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["curve_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate curve IDs.")
    if set(pd.to_numeric(frame["fold"], errors="raise").astype(int).unique()) != set(range(1, 6)):
        raise ValueError(f"{path} does not include frozen folds 1-5.")
    if frame.groupby("sanger_model_id")["fold"].nunique().gt(1).any():
        raise ValueError(f"{path} assigns a Sanger model to multiple folds.")
    numeric_columns = sorted(required - {"curve_id", "sanger_model_id"})
    numeric = frame[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{path} contains non-finite fold IDs, targets, or predictions.")
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    if reference is not None:
        left = reference.sort_values("curve_id").reset_index(drop=True)
        right = frame.sort_values("curve_id").reset_index(drop=True)
        if not left["curve_id"].equals(right["curve_id"]):
            raise ValueError(
                f"OOF curve IDs differ in {path}; repeated-seed results are not comparable."
            )
        for column in ("sanger_model_id", "fold"):
            if not left[column].astype(str).equals(right[column].astype(str)):
                raise ValueError(f"{column} assignments differ in {path}.")
        for column in ("actual_ln_ic50", "actual_auc"):
            if not np.allclose(
                left[column].to_numpy(dtype=float),
                right[column].to_numpy(dtype=float),
                rtol=0,
                atol=1e-8,
            ):
                raise ValueError(f"Observed {column} values differ in {path}.")
    return frame


def resolve_reference_run(
    reference: Path,
    *,
    fresh_run_set: bool,
    require_reference: bool,
) -> tuple[list[Path], pd.DataFrame | None]:
    """Return the prior reference only when the caller explicitly permits reuse."""
    if fresh_run_set:
        if require_reference:
            raise ValueError("--fresh-run-set cannot be combined with --require-reference.")
        return [], None
    existing_runs = [reference] if reference.is_file() else []
    reference_frame = validate_oof_file(reference) if existing_runs else None
    if require_reference and reference_frame is None:
        raise FileNotFoundError(
            f"Reference OOF predictions are required: {reference}"
        )
    return existing_runs, reference_frame


def run_command(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main(args) -> dict:
    data_dir = Path(args.data_dir).expanduser()
    graph_file = Path(args.graph_file).expanduser()
    output_root = Path(args.output_root).expanduser()
    reference = Path(args.reference_oof).expanduser()
    for path in (data_dir / "gdsc2_matched_response_pairs.csv",
                 data_dir / "gdsc2_matched_expression_features.npz",
                 graph_file, RUNNER, ANALYZER):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    existing_runs, reference_frame = resolve_reference_run(
        reference,
        fresh_run_set=args.fresh_run_set,
        require_reference=args.require_reference,
    )

    seeds = list(args.seeds)
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seed values must be unique.")
    if existing_runs and args.reference_seed in seeds:
        raise ValueError(
            f"Seed {args.reference_seed} is already represented by --reference-oof; "
            "do not train it a second time in this run."
        )

    generated_paths: list[Path] = []
    for seed in seeds:
        seed_dir = output_root / f"seed_{seed}"
        output_file = seed_dir / "gdsc2_pair_gnn_ablation_oof_predictions.csv"
        if output_file.exists() and not args.overwrite:
            raise FileExistsError(
                f"{output_file} already exists. Use --overwrite to replace that seed output."
            )
        command = [
            sys.executable, str(RUNNER),
            "--data-dir", str(data_dir),
            "--graph-file", str(graph_file),
            "--output-dir", str(seed_dir),
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--top-k-genes", str(args.top_k_genes),
            "--hidden-dim", str(args.hidden_dim),
            "--dropout", str(args.dropout),
            "--validation-fraction", str(args.validation_fraction),
            "--learning-rate", str(args.learning_rate),
            "--weight-decay", str(args.weight_decay),
        ]
        if args.overwrite and seed_dir.exists():
            # The training script overwrites named outputs; unrelated files in the
            # destination are left untouched. No automatic recursive deletion.
            print(f"Overwriting generated files for seed {seed} in {seed_dir}", flush=True)
        run_command(command)
        validated = validate_oof_file(output_file, reference_frame)
        if reference_frame is None:
            reference_frame = validated
        generated_paths.append(output_file)

    all_runs = existing_runs + generated_paths
    if not all_runs:
        raise ValueError("No reference or newly trained OOF runs were supplied.")

    analyzer_command = [
        sys.executable, str(ANALYZER),
        "--run", *[str(path) for path in all_runs],
        "--output-dir", str(output_root / "uncertainty"),
        "--bootstrap-replicates", str(args.bootstrap_replicates),
        "--seed", str(args.bootstrap_seed),
    ]
    run_command(analyzer_command)

    summary_path = output_root / "multiseed_run_manifest.json"
    summary = {
        "status": "completed",
        "reference_run": str(reference) if existing_runs else None,
        "reference_seed": args.reference_seed if existing_runs else None,
        "fresh_run_set": bool(args.fresh_run_set),
        "trained_seeds": seeds,
        "additional_seeds_trained": seeds if existing_runs else [],
        "oof_runs_included": [str(path) for path in all_runs],
        "all_runs_validated_same_curve_ids_folds_and_observed_targets": True,
        "same_outer_split": "frozen five-fold GroupKFold by Sanger model ID; folds 1-5",
        "training_configuration": {
            "epochs": args.epochs,
            "patience": args.patience,
            "top_k_genes": args.top_k_genes,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "validation_fraction": args.validation_fraction,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "uncertainty": {
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "output_directory": str(output_root / "uncertainty"),
        },
        "note": (
            "The original seed-42 run is included only if the reference OOF file exists. "
            "Additional seeds train separate models using unchanged outer fold IDs. "
            "Inner validation groups vary by seed; each remains disjoint from the outer test groups."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/ai/response_dataset")
    parser.add_argument("--graph-file", type=Path, default=ROOT / "data/processed/graph_v2/typed_graph_v2.pt")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/ai/gdsc2_pair_gnn_multiseed")
    parser.add_argument("--reference-oof", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--reference-seed", type=int, default=42)
    parser.add_argument("--require-reference", action="store_true")
    parser.add_argument("--fresh-run-set", action="store_true", help="Ignore all prior OOF files and train every supplied seed from scratch.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--top-k-genes", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=4201)
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
