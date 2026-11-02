# Pediatric-cancer model readiness audit

**Scope:** Determine whether the repository currently supports a pediatric-cancer response model, and identify the minimum next step without treating general-cancer or illustrative evidence as pediatric validation.

## Decision

**A small pediatric-origin single-agent cell-line model has now been trained and evaluated; a clinical-response model cannot be claimed.** There is no verified pediatric combination-response label set in the repository. The four-row file under `data/raw/gdsc_pediatric/` is illustrative, its provenance checks fail, and its values are not demonstrated to be measured outcomes.

An exploratory all-histology, known-drug single-agent expression model and a narrower pediatric-origin model have now been trained and evaluated. The narrower model uses Cellosaurus age records matched to Sanger model IDs to exclude lines with documented adult donors. Its results support a limited cell-line benchmark, not patient-level predictions, treatment recommendations, or clinical efficacy.

## Evidence found

| Resource | Observed evidence | Readiness implication |
|---|---|---|
| `data/raw/gdsc_pediatric/gdsc_pediatric_phase3.csv` and its manifest | Four rows; one drug-response row each for medulloblastoma, neuroblastoma, osteosarcoma, and acute lymphoblastic leukemia. The manifest says `GDSC2+PediatricResponse`, but supplies no source identifiers or measurement provenance. | Not a training set. |
| `data/ai/audits/evidence_provenance_gate.csv` | All four pediatric response rows are marked `NO_DIRECT_SOURCE_IDENTIFIER_FOUND`; the synthetic sample IDs do not exactly match OpenPedCan IDs. The audit explicitly says not to infer equivalence and that score/confidence is not assumed calibrated or experimentally measured. | These labels must be excluded until traced to original source records. |
| `data/raw/gdsc/gdsc2_responses.csv` and `data/raw/gdsc/gdsc2_manifest.json` | Official GDSC2 release dated 2023-10-27: 242,036 measured single-agent response rows, 969 cell lines, 295 drugs. Available outcomes include `ln_ic50` and `auc`. | Genuine response data, but single-agent and not intrinsically pediatric. |
| GDSC2 tumor-label subsets | `NB`: 32 cell lines / 7,469 response rows; `MB`: 4 / 1,072; `ALL`: 26 / 6,795. Each label has responses to most of the 295 drugs, but coverage varies by line. These are counts by GDSC2 reported cancer type, not verified age-at-diagnosis counts. | Candidate strata for a narrowly scoped, exploratory single-agent benchmark after identity/age review. Four MB lines are especially limited for independent generalization testing. |
| OpenPedCan v15 data | Histology metadata covers 47,895 biospecimens. The local manifests report somatic variants matched to 4,896 biospecimens, altered gene-level copy-number rows matched to 4,011, and fusion rows matched to 4,021. | Useful pediatric tumor molecular context, but no linked drug-response outcomes. OpenPedCan patient molecular observations cannot be joined to unrelated GDSC cell-line responses as if they were paired samples. |
| `data/raw/target/target_phase2.csv` | Five molecular-annotation rows with synthetic-looking `TARGET-00x` identifiers; the provenance gate finds no exact sample IDs in OpenPedCan and no direct source identifiers. | Exclude as validated patient molecular evidence until sourced and reconciled. |
| `data/raw/experimental/experimental_validation_phase7.csv` | Four qualitative drug/tumor claims with confidence-like values but no original study or record identifiers. The provenance gate marks them `NO_DIRECT_SOURCE_IDENTIFIER_FOUND`. | Not experimental validation evidence for training or evaluating a response model. |
| ALMANAC baseline | Measured combination responses from a general NCI-60 panel; the official score aggregation passed a stratified chart check, but full-dataset validation remains incomplete. | Technical combination benchmark only; not pediatric response evidence. |

## Single-agent expression benchmark run

The existing `scripts/ai/train_gdsc2_expression_model.py` pipeline completed on 2026-11-02. It trains one Ridge regression per known drug and endpoint from CCLE E-MTAB-2770 expression, using the existing five-fold split grouped by Sanger model ID. Variable-gene selection is performed within each training fold. The model was evaluated on 133,929 response rows covering 528 of 969 GDSC2 cell lines and all 295 drugs, matched using case-insensitive exact cell-line names.

| Endpoint | Expression Ridge MAE | Drug+cancer median MAE on same rows | Macro-drug MAE difference |
|---|---:|---:|---:|
| GDSC2 `LN_IC50` | 0.9860 | 1.0493 | -0.0668 |
| GDSC2 `AUC` | 0.0520 | 0.0514 | +0.0004 |

Lower MAE is better. Expression improves the pooled and macro-drug `LN_IC50` error modestly over this response-only baseline, while pooled `AUC` performance is essentially tied and its macro-drug error is slightly worse. This is exploratory validation on cell lines, not clinical or unseen-drug validation.

Held-out pediatric-associated histology results were evaluated only for cell lines matched to expression:

| GDSC2 label | Matched held-out lines / all lines in label | Matched response rows | `LN_IC50` Ridge MAE vs drug+cancer median | AUC Ridge MAE vs drug+cancer median |
|---|---:|---:|---:|---:|
| `NB` | 11 / 32 | 2,517 | 0.9554 vs 1.0255 | 0.0544 vs 0.0591 |
| `MB` | 2 / 4 | 557 | 0.8122 vs 0.7379 | 0.0415 vs 0.0389 |
| `ALL` | 13 / 26 | 3,383 | 0.7489 vs 1.2091 | 0.0583 vs 0.0737 |

The `MB` result is based on only two held-out matched lines and does not establish generalization. `ALL`, `NB`, and `MB` are GDSC2 cancer labels, not verified age/origin. The all-histology model includes training data from non-pediatric-associated cancer labels. These subset metrics support further evaluation, not a pediatric-specific performance claim.

The generated, ignored artifacts are `data/ai/molecular_models/gdsc2_expression_ridge.json`, the endpoint-specific out-of-fold prediction CSVs, and `data/ai/molecular_models/gdsc2_expression_ridge_summary.json`. The bundle predicts only GDSC2 response estimates for known drugs from compatible cell-line TPM inputs; it is not a patient-response or treatment-selection tool.

### Histology-restricted single-agent model

The same pipeline can be restricted to `NB`, `MB`, and `ALL` GDSC2 response labels. This scoped model has been trained with the command documented in `README.md`. Exact-name expression matching yielded 26 cell lines (11 NB, 2 MB, 13 ALL) and 6,457 measured response rows. Five-fold evaluation grouped by Sanger model ID; because the scoped dataset is small, fold training sizes ranged from 18 to 23 cell lines. Predictions were produced for 2,376 held-out rows across 183 known drugs.

| Endpoint | Expression Ridge MAE | Drug+cancer median MAE on same rows | Macro-drug MAE difference |
|---|---:|---:|---:|
| GDSC2 `LN_IC50` | 0.8342 | 0.9027 | -0.0697 |
| GDSC2 `AUC` | 0.0589 | 0.0589 | +0.0005 |

This narrowly scoped evaluation is closer to the intended pediatric-associated context than the pan-cancer benchmark, but it does not prove pediatric origin, has limited cell-line coverage, and remains exploratory. The model was trained across the three reported tumor labels together, not as separate NB, MB, and ALL predictors. The result should not be used to prioritize patient treatment or interpreted as clinical efficacy.

Generated, ignored scoped artifacts:
- `data/ai/molecular_models/gdsc2_pediatric_associated_expression_ridge.json`
- `data/ai/molecular_models/gdsc2_pediatric_associated_expression_ridge_summary.json`
- `data/ai/molecular_models/gdsc2_pediatric_associated_expression_ridge_ln_ic50_oof_predictions.csv`
- `data/ai/molecular_models/gdsc2_pediatric_associated_expression_ridge_auc_oof_predictions.csv`

### Pediatric-origin single-agent model

The Cellosaurus API audit is reproducible with `scripts/ai/audit_gdsc2_model_origins.py`. For all 26 expression-matched GDSC2 lines, the exact Sanger Cell Model Passport ID was found in the Cellosaurus record. Cellosaurus recorded donor ages under 18 for 23 lines and ages of 27, 38, and 28 years for three ALL-labeled lines, which were excluded. The 23 included ages were NB 11/11 under 18, MB 2/2 under 18, and ALL 10/13 under 18. The source-backed row-level crosswalk is `reports/gdsc2_pediatric_model_origin_audit.csv`; each row includes the Cellosaurus accession, recorded age, disease labels, Cellosaurus URL, and Sanger Cell Model Passport URL.

The model was retrained on the 23 pediatric-origin-confirmed GDSC2 lines across `NB`, `MB`, and `ALL`. Exact-name expression matching covered all 23, providing 5,719 response rows. It was evaluated with leave-one-Sanger-model-ID-out validation, training on 22 lines per fold. Predictions were available for 2,896 held-out response rows across 128 known drugs.

| Endpoint | Pediatric-origin expression Ridge MAE | Drug+cancer median MAE on same rows | Macro-drug MAE difference |
|---|---:|---:|---:|
| GDSC2 `LN_IC50` | 0.8605 | 0.9337 | -0.0734 |
| GDSC2 `AUC` | 0.05994 | 0.06010 | -0.00019 |

Both endpoints show modestly lower pooled MAE than the response-only drug+cancer median baseline in this leave-one-line-out evaluation. However, the model has only 23 independent lines, few lines per tumor group, uses one shared model across three histologies, and lacks external pediatric assay validation. Per-label held-out rows show the `AUC` model does not consistently improve the response-only baseline, especially in the two-line MB subset. The 128-drug results apply only to drugs meeting the model's training threshold. Do not interpret these values as patient-response performance.

The audit and pediatric-origin training commands are documented in `README.md`. Generated, ignored model artifacts:
- `data/ai/molecular_models/gdsc2_pediatric_origin_verified_expression_ridge.json`
- `data/ai/molecular_models/gdsc2_pediatric_origin_verified_expression_ridge_summary.json`
- `data/ai/molecular_models/gdsc2_pediatric_origin_verified_expression_ridge_ln_ic50_oof_predictions.csv`
- `data/ai/molecular_models/gdsc2_pediatric_origin_verified_expression_ridge_auc_oof_predictions.csv`

## What can be built now

1. **ALMANAC combination baseline:** already exists as a general-cancer technical benchmark. It should remain explicitly non-pediatric and exploratory.
2. **GDSC2 single-agent expression models:** trained both pan-cancer and on a Cellosaurus age-confirmed pediatric-origin subset. The latter is a small cell-line model, not clinically validated.
3. **Pediatric molecular evidence layer:** OpenPedCan provides disease and molecular observations, but they are not treatment-response labels. It can inform feature engineering and biological interpretation only after appropriate sample/feature curation; it cannot supervise response prediction by itself.

## Remaining boundary and next step

The minimum data prerequisite for a small pediatric-origin single-agent cell-line benchmark is now met: measured GDSC2 response labels, expression features, and age-verified cell-line identities overlap for 23 lines. The remaining limitation is model-validation strength and transfer to patient decisions, not the presence of training labels. Further useful work is independent validation on additional age-verified pediatric models or assays, and disease-specific evaluation once adequate samples are available.

For a **pediatric combination-response model**, the same requirements apply to measured combination labels, including both drug identities, dose/assay details, and a defined combination-response endpoint. No such pediatric combination dataset is currently verified in this repository. Adult/general ALMANAC scores must not be relabeled as pediatric outcomes.

## Next steps

1. **Keep the four illustrative pediatric rows, synthetic TARGET rows, and unsupported phase-7 claims out of training and validation.**
2. **Treat the pediatric-origin single-agent model as an exploratory cell-line model**, limited to 23 models and 128 drugs; do not use it to predict patient response or recommend treatment.
3. **When resources allow, independently validate** against additional source-backed pediatric-derived models or experimental response data.
4. **At the later combination stage,** acquire and curate source-backed pediatric combination measurements; do not relabel adult/general ALMANAC outcomes as pediatric.

**Bottom line:** the single-agent model is now trained on measured responses from 23 cell lines with Cellosaurus-documented donor ages under 18, using exact Sanger model-ID matching and leave-one-line-out evaluation. It is not a clinical predictor. Combination modeling remains a separate later-stage task requiring its own pediatric response labels.
