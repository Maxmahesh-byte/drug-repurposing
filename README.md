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