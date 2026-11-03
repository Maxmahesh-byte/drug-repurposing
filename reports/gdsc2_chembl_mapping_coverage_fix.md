# GDSC2–ChEMBL mapping coverage correction

## Motivation

The controlled evidence-layer experiment asks whether progressively adding (1) a drug-to-ChEMBL molecule mapping, (2) a direct molecule-to-target mechanism, and (3) a target-to-gene-symbol annotation improves prediction.

The earlier ingestion export, `gdsc2_chembl_target_evidence.csv`, was built from direct human-target mechanism rows. The graph builder also created the drug-to-molecule mapping edges from that same table. Consequently, a uniquely mapped GDSC2 drug without a qualifying direct human mechanism did not receive a mapping edge.

The current importer audit reported 167 unique GDSC2-to-ChEMBL mappings and 97 drugs with direct human mechanisms. The mapping layer was therefore narrower than the intended entity-resolution layer.

## Change

The ingestion pipeline now writes two independent CSV files:

- `gdsc2_drug_molecule_mappings.csv`: one row per uniquely resolved GDSC2 drug and ChEMBL molecule, including exact-match method, matched synonym, synonym type, ChEMBL molecule URL, and provenance state.
- `gdsc2_chembl_target_evidence.csv`: direct human mechanism and target-component gene-symbol evidence only.

A uniquely mapped molecule is retained even when the target-evidence table contains no row for it. Ambiguous matches remain excluded. Exact preferred-name and exact synonym matching remain literal string-resolution policies and are not treated as proof of chemical equivalence.

The graph builder now creates:

- GDSC drug → ChEMBL molecule edges from the mapping table;
- ChEMBL molecule → target edges from direct human mechanism evidence;
- target → gene-symbol edges from ChEMBL target-component annotations.

The graph can therefore represent mapping without asserting a target mechanism. An empty target-evidence CSV with headers is valid when molecule mappings exist but no direct human target mechanism qualifies.

## Reproducible rebuild

After pulling this branch, ensure the local OpenPedCan, GDSC2, and expression inputs used by the project are available. Rebuild the source tables and graph:

```bash
.venv/bin/python -c 'from src.ingestion.gdsc2 import build_gdsc2_response_dataset; build_gdsc2_response_dataset()'
.venv/bin/python -c 'from src.ingestion.chembl_targets import build_chembl_target_dataset; build_chembl_target_dataset()'
.venv/bin/python src/preprocessing/build_graph.py
.venv/bin/python scripts/migrate_graph_v2.py
```

The first graph build also requires the prepared OpenPedCan Phase 1–4 files. The typed graph output used by the pair-GNN is `data/processed/graph_v2/typed_graph_v2.pt`.

Run focused tests:

```bash
.venv/bin/python -m pytest tests/test_chembl_targets.py tests/test_gdsc2_pair_gnn.py tests/test_gdsc2_pair_gnn_ablations.py tests/test_gdsc2_pair_gnn_multiseed.py -q
```

Then train a completely fresh five-seed experiment into a new directory:

```bash
.venv/bin/python scripts/ai/run_gdsc2_pair_gnn_multiseed.py \
  --fresh-run-set \
  --seeds 42 43 44 45 46 \
  --output-root data/ai/gdsc2_evidence_layer_ablations_full_mapping
```

Do not overwrite or relabel the earlier `data/ai/gdsc2_evidence_layer_ablations` outputs. Those results describe the prior, restricted mapping graph and remain useful as a historical comparison, but they are not the corrected full-mapping experiment.

## Interpretation limits

This change corrects the intended coverage of the ChEMBL mapping relation; it does not prove that every exact-name mapping is chemically equivalent, that a ChEMBL mechanism is active in a specific cell line, or that target evidence is causal. The corrected five-seed experiment must be run before drawing conclusions about the predictive contribution of the expanded mapping layer.
