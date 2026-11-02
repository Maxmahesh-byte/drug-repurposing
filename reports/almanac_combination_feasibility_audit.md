# NCI-ALMANAC combination-data feasibility audit

**Audit date:** 2026-11-02
**Scope:** Feasibility of using measured combination outcomes to develop a combination-response capability for pediatric-cancer drug repurposing. Neuroblastoma is an exploratory context, not the target population.

## Summary

The public NCI-ALMANAC archive is accessible from an NCI-hosted download and contains measured combination responses suitable for a separate, general combination-response modeling task. It is not a pediatric-cancer-specific screen. The project has no substantive pediatric combination-response dataset: `data/raw/gdsc_pediatric/gdsc_pediatric_phase3.csv` contains four example rows, while GDSC2 contains single-agent responses.

Therefore, ALMANAC can support development and internal evaluation of a general combination model, but cannot by itself establish performance in pediatric cancers. Its labels must not be joined to pediatric cell lines solely by similar names or treated as single-agent response labels.

## Source and observed schema

- Downloaded the NCI-hosted archive [`ComboDrugGrowth_Nov2017.zip`](https://wiki.nci.nih.gov/download/attachments/338237347/ComboDrugGrowth_Nov2017.zip) and its official compound-name crosswalk to the ignored `data/raw/almanac/` cache for reproducibility. The archive contains `ComboDrugGrowth_Nov2017.csv` (615,194,859 uncompressed bytes); raw inputs are not version controlled.
- The raw CSV has 3,686,475 dose-level rows and 29 columns. Relevant fields include `NSC1`, `NSC2`, `CONC1`, `CONC2`, `PERCENTGROWTH`, `EXPECTEDGROWTH`, `SCORE`, `VALID`, `PANEL`, and `CELLNAME`.
- All 3,686,475 rows in this copy had `VALID=Y`.
- Removing non-printing control characters yields 60 distinct cell-line names. Of the 3,686,475 valid dose-level rows, 812,961 have a missing compound identifier and are single-agent/control or incomplete-identity records; they are excluded from combination analyses. The remaining 2,873,514 rows have both NSC identifiers.
- These are dose-level measurements, not independent pair-level training examples. Before training, the dose grid and replicates must be processed according to the ALMANAC scoring method; the raw per-dose `SCORE` must not be treated as an already aggregated ComboScore without reproducing and validating the aggregation.
- The official compound-name crosswalk contains 105 NSC identifiers resolving to 104 canonical active-drug names. After canonicalizing aliases, there are 5,233 unordered pairs including one same-active-drug pair caused by two NSC identifiers for vemurafenib; excluding that pair leaves 5,232 distinct unordered drug pairs, matching the publication. The canonicalized archive contains 304,549 cell-line/pair groups before removing same-drug records, 304,489 afterward, and 304,423 after dose-reversal QC. The 60-group self-pair difference corresponds to vemurafenib across the 60 cell lines; reversal QC then excludes 66 additional groups.
- The chunked ingestion applies the published adjacent-dose reversal criterion to both drug axes, comparing mean `PERCENTGROWTH` at each dose-index point within the same cell line, unordered pair, screener, and study. It flags a reversal when growth rises by at least 50 percentage points between adjacent indices. In the complete archive, 76 adjacent-dose comparisons trigger the threshold across 66 cell-line/pair groups; those groups are excluded.
- After reversal QC, 138 of 304,423 cell-line/pair groups have incomplete dose-level `SCORE` values. The pipeline leaves those group scores missing rather than imputing zero. Of the 2,872,974 non-self combination dose rows, 2,070 lack both `EXPECTEDGROWTH` and row-level `SCORE`; no combination rows lack `PERCENTGROWTH`.
- Official chart reconciliation established the score aggregation: within each cell-line/pair/screener/study, average replicate row-level `SCORE` values at the same dose point, sum the dose-point means, scale the result to a nine-point equivalent (`9 / number of dose points`), truncate toward zero, then sum study scores. This corrects the former direct sum, which overestimated 15-point panels, and handles duplicate rows in some 9-point assays. Drug-pair ordering is now case-insensitive and shared by ingestion and validation.
- The deterministic score-stratified validation sampled 100 uniquely NSC-mapped pairs (20 per each of five strata) and fetched all 100 official detailed charts. All 5,829 cell-line/pair scores for which an official score was available matched exactly, including seven groups intentionally excluded from the modeling output by dose-reversal QC; there were zero score mismatches, incomplete local scores, or missing local groups. Two complete local groups for Methoxsalen+Topotecan hydrochloride (MOLT-4 and SR) have no corresponding score on the official chart. These are recorded as official-chart coverage gaps, not score mismatches. Thus the sampled score-value validation passed, while chart coverage is not complete and the full dataset has not been exhaustively checked. The unrounded `EXPECTEDGROWTH - PERCENTGROWTH` sum remains a diagnostic; its rounded value differs from row-level `SCORE` in 5,772 comparable non-self dose rows.
- The DTP all-pair mean heatmap is not a direct check of the mean of our per-cell-line sums. For Anastrozole+Azacitidine, the heatmap shows -16, while the mean of local per-cell-line sums is -135; the per-pair detailed chart gives 116 study/cell-line scores whose mean is -68.9 and whose per-cell-line maximum has mean -15.9. Thus the heatmap uses a different aggregation unit and is not used as a grouped-label reference.
- The published ALMANAC method defines positive ComboScores as greater-than-additive activity using modified Bliss independence. The score aggregation now matches the available official chart scores in the stratified 100-pair validation; `training_ready` remains false because the full dataset is not exhaustively validated, two local groups lack official chart values, and identity/transfer limitations remain. The public CANDLE/Combo example also loads the same raw ALMANAC archive and its `use_combo_score` option takes a maximum dose-level `SCORE`; that is a different target, not an independent reproduction of the grouped score.

## Relevance and overlap with local data

- The NCI-ALMANAC panels in the downloaded file are colon, leukemia, non-small-cell lung, CNS, renal, melanoma, prostate, ovarian, and breast cancer. This is a general cancer-cell-line panel, not a pediatric-cancer cohort.
- Thirty-eight ALMANAC cell-line names exactly matched GDSC2 names case-insensitively. Two additional names matched after removing an `/ATCC` suffix (`A549/ATCC` and `MDA-MB-231/ATCC`). These are candidate name-based overlaps only; stable cell-line identifiers and assay provenance still need confirmation before merging features.
- Exact case-insensitive drug-name matching identifies 25 of the 104 ALMANAC drug names in GDSC2. Twenty-one map to one GDSC2 drug ID and four map to multiple IDs; the other 79 have no exact-name candidate. These are candidates, not validated chemical-identity links, and ambiguous IDs are not selected automatically.
- Those 38 candidates span multiple GDSC2 cancer labels. Two are labeled `ALL`; the rest include adult-associated disease labels, other leukemia labels, and solid tumors. This does not constitute a pediatric cohort or pediatric validation.
- Twenty-three of the 38 exact GDSC2 name matches appear among models with matched E-MTAB-2770 expression profiles in the existing bortezomib expression report. Including the two `/ATCC`-normalized aliases, 25 of the 60 ALMANAC lines have a matching expression profile in that report. This confirms some expression-feature overlap, not full molecular coverage or a validated cross-study identity mapping.
- The local GDSC2 file contains single-agent outcomes for 969 cell lines. These are useful as a separate task and possible molecular-data source, but cannot supply combination-response labels.
- The local pediatric response CSV contains only four illustrative rows; its manifest identifies the source as `GDSC2+PediatricResponse`, which does not establish a curated, independently measured pediatric combination dataset.

## Modeling implications

1. Keep the combination model separate from the single-agent GDSC2 model.
2. Resolve and document drug identity from ALMANAC NSC identifiers before matching compounds to project drug records or molecular descriptors.
3. The 100-pair score-stratified official-chart check passed for all 5,829 available score comparisons. Two chart values are unavailable (Methoxsalen+Topotecan hydrochloride in MOLT-4 and SR); the local source rows are complete, but these labels lack independent chart confirmation. The complete dataset remains unvalidated, so preserve `training_ready=false` until the remaining required validation and identity checks are resolved.
4. Link cell lines to molecular features using stable identifiers where possible; treat name-only matches as candidates pending review.
5. Compare simple baselines with models using grouped validation that holds out drug pairs and cell lines. Random row splits would leak highly related dose measurements across splits.
6. Describe results as general cell-line combination-response modeling. Pediatric-cancer transfer remains unvalidated until suitable pediatric combination measurements are available.

## Exploratory ALMANAC-only baseline

An initial five-fold Ridge regression baseline was trained on the 304,285 complete, reversal-filtered cell-line/drug-pair scores. It uses only one-hot drug identities and cell-line identity; it has no molecular features or explicit drug-pair interaction terms. All out-of-fold predictions hold out the entire group relevant to the evaluation:

| Evaluation split | Baseline | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Hold out drug pairs | Global mean | 42.41 | 65.25 | -0.000 |
| Hold out drug pairs | Cell-line mean | 42.11 | 64.77 | 0.014 |
| Hold out drug pairs | Drug + cell-line additive Ridge | 40.92 | 62.52 | 0.082 |
| Hold out cell lines | Global mean | 42.43 | 65.26 | -0.001 |
| Hold out cell lines | Drug-pair mean from other cell lines | 35.45 | 51.82 | 0.369 |
| Hold out cell lines | Drug-additive Ridge | 41.08 | 62.68 | 0.077 |

These are internal NCI-60 cell-line results, not pediatric-cancer validation. The drug-pair mean performs substantially better when the same pair has observations in other cell lines, while the drug-identity additive model has only modest performance for unseen combinations. This is evidence that pair-specific effects matter and that this baseline alone is not sufficient for prioritizing novel combinations. The final Ridge model, out-of-fold predictions, and metrics are generated under `artifacts/almanac_baseline/`; the model is exploratory and does not change `training_ready=false`.

## Sources

- [NCI-hosted ALMANAC raw-data archive](https://wiki.nci.nih.gov/download/attachments/338237347/ComboDrugGrowth_Nov2017.zip)
- [Holbeck et al., NCI-ALMANAC publication](https://pmc.ncbi.nlm.nih.gov/articles/PMC5499996/)
- [NCI/DOE combination-prediction resource and processed modeling subset](https://github.com/CBIIT/NCI-DOE-Collab-Pilot1-Combo-combination-drug-response-predictor)
- [Xia et al., NCI-ALMANAC combination-response model](https://pmc.ncbi.nlm.nih.gov/articles/PMC6302446/)
- [NCI-ALMANAC results and download page](https://dtp.cancer.gov/ncialmanac/initializePage.do)
- [Official Anastrozole+Azacitidine detailed scores](https://dtp.cancer.gov/ncialmanac/getScoresChart.do?nsc1=102816&nsc2=719344)
- [Official Bortezomib+Vorinostat detailed scores](https://dtp.cancer.gov/ncialmanac/getScoresChart.do?nsc1=681239&nsc2=701852)
- [Official Cisplatin+Paclitaxel detailed scores](https://dtp.cancer.gov/ncialmanac/getScoresChart.do?nsc1=119875&nsc2=125973)
- Local data inspected: `data/raw/gdsc/gdsc2_responses.csv`, `data/raw/gdsc/gdsc2_manifest.json`, `data/raw/gdsc_pediatric/gdsc_pediatric_phase3.csv`, `data/raw/gdsc_pediatric/gdsc_pediatric_manifest.json`, and `reports/bortezomib_proteasome_expression_models.csv`.
