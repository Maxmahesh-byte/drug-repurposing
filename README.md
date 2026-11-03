# drug-repurposing
using kg and gnn to find better drug protein paths
# MLOps Pipeline for Cancer Drug Repurposing using Heterogeneous Knowledge Graphs and GNNs

## Overview
End-to-end MLOps pipeline for predicting new cancer indications for existing drugs using Knowledge Graphs + GNNs.

## How to Run
```bash
docker compose up --build
```

## OpenPedCan Data

Phase 1 reads the official `histologies.tsv` clinical metadata table from an OpenPedCan release. Obtain a release using the [upstream download instructions](https://github.com/d3b-center/OpenPedCan-analysis#how-to-obtain-openpedcan-data), then point this project at the downloaded table:

```bash
export OPENPEDCAN_DATA_PATH="/path/to/OpenPedCan-analysis/data/histologies.tsv"
.venv/bin/python -c 'from src.ingestion.openpedcan import build_phase1_dataset; build_phase1_dataset()'
```

The loader fails if the release file is missing; it does not silently substitute sample data. For development-only fixtures, set `OPENPEDCAN_ALLOW_SYNTHETIC_FALLBACK=1`. Phase 1 contains clinical cohort, diagnosis, and subtype metadata; molecular alterations come from the separate consensus MAF file.

Phase 2 reads the full-release `snv-consensus-plus-hotspots.maf.tsv.gz` and joins `Tumor_Sample_Barcode` to the unique biospecimen ID in Phase 1. Each variant row also receives its real cohort, tumor type, and diagnosis metadata; unmatched barcodes are excluded and counted in the manifest. Set `OPENPEDCAN_MAF_PATH` when the file is outside `data/raw/openpedcan/`. The smaller OpenPedCan testing subset is retained for bounded tests only; full-release files are preferred automatically when both are present.

Phase 3 ingests `consensus_wgs_plus_cnvkit_wxs_plus_freec_tumor_only.tsv.gz`, excludes neutral calls, and joins `biospecimen_id` to the clinical table. Phase 4 ingests `fusion-putative-oncogenic.tsv` and joins its `Sample` barcode. Both outputs retain cohort, tumor type, and diagnosis annotations. Their source locations and match counts are recorded in separate manifests. In the graph, copy-number calls are aggregated by tumor type and gene to keep memory bounded; fusion evidence remains linked to its biospecimen.

## GDSC2 Drug Response

The response layer uses the official [Wellcome Sanger GDSC2 fitted dose-response release](https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_27Oct23.xlsx). Place `GDSC2_fitted_dose_response_27Oct23.xlsx` under `data/raw/gdsc/`, or set `GDSC2_DATA_PATH`. The verified workbook MD5 is `a57b46470f325ae5800dca24c72c19f5`. The importer preserves Sanger model IDs, cell-line names, reported `TCGA_DESC`, drug IDs, `LN_IC50`, and `AUC`; it does not map cell lines to OpenPedCan patients or harmonize cancer labels implicitly.
`PUTATIVE_TARGET` and `PATHWAY_NAME` are retained as source-provided annotations, not treated as validated drug-target evidence.

## ChEMBL Target Evidence

The target importer resolves GDSC2 drug names by exact ChEMBL preferred-name or exact synonym match, then keeps direct mechanisms for human targets and gene symbols from ChEMBL target-component metadata. Unmatched and ambiguous names, plus exact matches without a direct human mechanism, are recorded in the manifest rather than guessed. In the current Sanger release, 167 of 295 drug names map uniquely; 97 have direct human mechanisms, 70 have no direct mechanism evidence, 126 remain unmatched, and 2 are ambiguous.

The active DVC pipeline ingests four real OpenPedCan evidence layers, GDSC2 cell-line responses, and ChEMBL direct human-target mechanisms. Node features are deterministic entity types; edge features retain measured molecular counts and GDSC2 responses. In the OpenPedCan graph, copy-number calls are summarized by tumour type and gene, with gain/amplification counts, loss/deletion counts, and unclassified statuses represented as separate named edge attributes; the original status-specific counts remain in edge provenance. These are cohort summaries, not sample-level calls. GNN training and repurposing scores remain disabled until a validated drug-response objective and justified transfer from cell-line models to pediatric disease are implemented. No generated labels or fallback scores are used.

### Exploratory expression-response benchmark

`scripts/ai/train_gdsc2_expression_model.py` builds per-drug Ridge regressions for GDSC2 `LN(IC50)` and AUC using the public CCLE E-MTAB-2770 RNA-seq matrix:

```bash
./.venv/bin/python scripts/ai/train_gdsc2_expression_model.py
```

To use a previously downloaded local copy of the same TPM matrix and record its file checksum in the model bundle and summary:

```bash
./.venv/bin/python scripts/ai/train_gdsc2_expression_model.py \
  --expression-file data/raw/expression/E-MTAB-2770_tpmss.tsv
```

The local file uses the same validation, `log2(TPM + 1)` transformation, and fold-specific feature selection as the streamed source. Keep large expression matrices out of Git and retain their SHA-256 checksums in local provenance records.

At the initial audit, case-insensitive exact cell-line names matched 528 of 969 GDSC2 models; ambiguous matches are excluded. The script reuses the frozen five-fold Sanger-ID held-out-cell-line assignments, selects variable genes using each training fold only, and compares predictions against response-only baselines on the same matched rows. It streams the full public expression matrix, so the first run may take substantial time. The generated JSON bundle can be used with `src.inference.gdsc2_expression.predict_expression_response`; input values must be raw TPM keyed by the same gene IDs. This is limited to known drugs and cell-line response estimates; it does not predict unseen drugs, patient response, clinical efficacy, or combinations.

The Streamlit dashboard includes a separate **Molecular response model** workspace for running the benchmark, reviewing results, and optionally estimating response from an uploaded cell-line TPM profile after a trained model bundle exists.

For an explicitly scoped benchmark using only cell lines whose source-donor age is documented as under 18 in Cellosaurus and whose Sanger model IDs match GDSC2:

```bash
./.venv/bin/python -m scripts.ai.audit_gdsc2_model_origins
./.venv/bin/python scripts/ai/train_gdsc2_expression_model.py \
  --cancer-types NB MB ALL \
  --model-origin-audit reports/gdsc2_pediatric_model_origin_audit.csv \
  --output-prefix gdsc2_pediatric_origin_verified_expression_ridge
```

The origin audit queries the Cellosaurus API and verifies the stable Cell Model Passport ID cross-reference. The model uses leave-one-Sanger-model-ID-out evaluation. Here, "pediatric-origin verified" means Cellosaurus documents a source-donor age under 18; it does not establish clinical response prediction. The model remains a small cell-line benchmark for known drugs, not a patient-response or treatment-selection tool. The origin evidence and scope are recorded with the results.

Combination-response modeling remains separate: no combination/synergy response dataset is currently present in `data/`. Such a model requires measured drug-pair outcomes and pair-aware validation; single-agent labels are not treated as combination labels.
### Auditable E-MTAB-2770 to GDSC2 cell-line crosswalk

Build a local crosswalk from Expression Atlas sample labels to GDSC2 Sanger model IDs and the typed graph's cell-line nodes:

```bash
./.venv/bin/python scripts/ai/build_gdsc2_expression_crosswalk.py
```

The script reads only the expression matrix header, then maps one row per GDSC2 Sanger model using the same case-insensitive exact cell-line-name rule as the expression Ridge benchmark. It checks the expected graph node ID convention (`GDSC_MODEL:<sanger_model_id>`) against `graph_v2/typed_graph_v2.pt`. It does not guess aliases, normalise punctuation, or use tumour labels to force matches. Ambiguous and unmatched models remain in the output with explicit statuses.

Default outputs are written under `data/ai/audits/`:
- `gdsc2_expression_cell_line_crosswalk.csv`
- `gdsc2_expression_cell_line_crosswalk_summary.json`

Override the input or output locations with `--expression-file`, `--response-file`, `--graph-file`, `--output-file`, or `--summary-file`. The script records SHA-256 checksums for the input matrix and GDSC2 response file. Load only graph artifacts generated by your trusted local pipeline, because PyTorch graph deserialization uses pickle.


### Leakage-aware GDSC2 response dataset for GNN experiments

After the local expression crosswalk has been built and audited, prepare a standalone response-pair table and model-aligned expression matrix:

```bash
./.venv/bin/python scripts/ai/prepare_gdsc2_gnn_dataset.py \
  --expression-file data/raw/expression/E-MTAB-2770_tpmss.tsv \
  --crosswalk-file data/ai/audits/gdsc2_expression_cell_line_crosswalk.csv
```

The builder reuses the validated GDSC2 response loader and frozen five-fold assignments, checks every crosswalk sample label and graph cell-line ID, and excludes models without an exact expression match. It writes `data/ai/response_dataset/gdsc2_matched_response_pairs.csv`, a compressed expression-feature matrix, and a JSON provenance/validation summary. Expression rows are aligned to sorted Sanger model IDs; pair rows carry the corresponding expression-row index.

Crucially, measured `ln_ic50` and `auc` remain labels in the pair table, not node or edge features. The legacy graph's `gdsc2_response` relation must not be passed to message passing; alternatively, a rigorously fold-filtered training-only relation would be required. Fit expression scaling and feature selection inside each training fold only. This step prepares data; it does not train a GNN or establish clinical prediction.


### Five-fold drug–cell-line pair GNN benchmark

After preparing the response-pair dataset, run the exploratory pair model:

```bash
./.venv/bin/python -m pytest tests/test_gdsc2_pair_gnn.py -q

./.venv/bin/python scripts/ai/train_gdsc2_pair_gnn.py \
  --data-dir data/ai/response_dataset \
  --graph-file data/processed/graph_v2/typed_graph_v2.pt \
  --output-dir data/ai/gdsc2_pair_gnn
```

The model uses a compact drug-side RGCN over only the typed evidence relations drug→ChEMBL molecule→direct mechanism target→gene. Cell-line expression is encoded separately and fused with the drug embedding by a pairwise prediction head with LN(IC50) and AUC outputs. The legacy `gdsc2_response` relation is not copied into the model graph.

The five outer folds are the frozen GroupKFold assignments by Sanger model ID. An inner group-level validation split is taken only from each outer training set. Expression variance feature selection and standardisation are fitted only on the inner-fit cell lines for that outer fold. The script saves fold checkpoints, out-of-fold predictions, metrics, and a same-curve-ID comparison against the existing drug+cancer median baseline when the baseline files are present.

This is a first transductive known-drug benchmark. It is not an unseen-drug test, patient-outcome model, or clinical predictor. Review the saved fold metrics and same-row baseline comparison before interpreting any apparent gain.


### Controlled GDSC2 pair-model ablations

Run the matched five-fold ablation experiment after preparing the response-pair dataset and local graph artifact:

```bash
./.venv/bin/python -m pytest tests/test_gdsc2_pair_gnn_ablations.py -q

./.venv/bin/python scripts/ai/run_gdsc2_pair_gnn_ablations.py \
  --data-dir data/ai/response_dataset \
  --graph-file data/processed/graph_v2/typed_graph_v2.pt \
  --output-dir data/ai/gdsc2_pair_gnn_ablations
```

The runner compares three variants on identical frozen outer folds and inner cell-line validation groups:
- `full_graph`: RGCN drug-target evidence encoder + expression encoder + pair head.
- `drug_id_only`: learned drug identity embedding + expression encoder, with no graph propagation.
- `expression_only`: expression encoder only, with neither drug identity nor graph.

All variants share fold-specific expression feature selection/scaling. LN(IC50) and AUC are standardised independently using inner-fit response rows only; predictions are inverse-transformed before MAE, RMSE, and R² are reported. This avoids the larger numerical scale of LN(IC50) dominating the joint training loss. Fold checkpoints, OOF predictions, fold metrics, and external baseline comparisons (where exact curve-ID and fold overlap is available) are saved locally. The experiment remains a known-drug, held-out-cell-line, transductive in-vitro benchmark, not an unseen-drug or clinical evaluation.


### Repeated-seed pair-GNN ablations and uncertainty

To reuse the existing seed-42 ablation OOF output and train four additional seeds (43–46) on the same frozen outer folds:

```bash
./.venv/bin/python -m pytest tests/test_gdsc2_pair_gnn_multiseed.py -q

./.venv/bin/python scripts/ai/run_gdsc2_pair_gnn_multiseed.py \
  --seeds 43 44 45 46 \
  --reference-oof data/ai/gdsc2_pair_gnn_ablations/gdsc2_pair_gnn_ablation_oof_predictions.csv \
  --output-root data/ai/gdsc2_pair_gnn_multiseed
```

The orchestrator trains the full-graph, drug-ID-only, and expression-only variants for each additional seed. It checks that every OOF file contains the same response curve IDs, Sanger model IDs, frozen outer-fold assignments, and observed response labels as the reference seed. Inner validation groups and initialisation vary with seed, while the outer test folds stay fixed. It then invokes `analyze_gdsc2_pair_gnn_uncertainty.py` across all OOF runs and saves a manifest plus bootstrap results.

Each seed is written into a separate directory; existing outputs are not overwritten unless `--overwrite` is specified. A new run therefore does not replace the original seed-42 results. The cell-line cluster bootstrap remains conditional on each fitted model; repeated seeds are summarised separately to show sensitivity to training randomness.


### GDSC2 pair-GNN leakage and evaluation audit

The current audit changes and test coverage are documented in [reports/gdsc2_pair_gnn_leakage_audit.md](reports/gdsc2_pair_gnn_leakage_audit.md). The compact model graph uses only drug→ChEMBL molecule→direct target→gene evidence and excludes the GDSC2 response relation and its response-valued edge attributes. Three RGCN layers allow gene-node information to traverse the full reverse evidence path back to drug nodes.

The ablation runner now validates that every response pair's expression row index maps to its Sanger model ID, that its graph node ID is the expected namespaced cell-line ID, and that Sanger groups are disjoint across inner-fit, inner-validation, and outer-test partitions. Expression selection/scaling and separate endpoint target scaling are fitted on inner-fit data only. Test-row labels are not converted to training tensors. Baseline joins also validate curve IDs, Sanger IDs, frozen folds, and observed targets. The run summary reports how many drugs have a ChEMBL mapping, a direct target path, and a complete gene-level path.

**Evaluation caveat:** GNN gradient updates use the inner-fit subset, while existing Ridge and drug+cancer median baselines use the full outer-training fold. They share held-out curves and folds, but not the same amount of training data. A follow-up refit on the full outer-training fold after inner validation selects the training epoch would make the estimator comparison more closely matched. The five-seed metrics from the earlier two-layer model do not describe this revised three-layer architecture.


For a new architecture revision (for example, the audited three-layer encoder), do not reuse OOF predictions from the earlier two-layer model. Train a fresh seed set into a new output directory:

```bash
./.venv/bin/python scripts/ai/run_gdsc2_pair_gnn_multiseed.py \
  --fresh-run-set \
  --seeds 42 43 44 45 46 \
  --output-root data/ai/gdsc2_pair_gnn_multiseed_3layer
```

The `--fresh-run-set` option ignores the script's default seed-42 reference OOF file and trains all five supplied seeds from scratch. Keep this output separate from the previous two-layer `gdsc2_pair_gnn_multiseed` directory.
