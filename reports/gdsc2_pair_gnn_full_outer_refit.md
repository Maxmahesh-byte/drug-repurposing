# GDSC2 pair-GNN full outer-training refit

## Motivation

The earlier pair-GNN training procedure used an inner-fit subset for gradient updates and an inner-validation subset for epoch selection. The existing Ridge and drug+cancer median OOF baselines used the complete outer-training fold. Although the models shared held-out curves and frozen outer folds, the GNN had fewer training cell lines and response records.

## Updated procedure

For each frozen outer fold and each model variant:

1. Split outer-training cell-line groups into inner-fit and inner-validation subsets. The outer-test groups remain separate.
2. Train a temporary model on inner-fit rows. Fit expression feature selection/scaling and separate LN(IC50)/AUC target scaling on inner-fit data. Use inner-validation loss only to select the epoch count.
3. Discard the epoch-selection model.
4. Reinitialise a fresh model of the same variant.
5. Refit expression variance selection, expression mean/std scaling, and the two endpoint target scalers on the complete outer-training fold.
6. Train the fresh model for the selected number of epochs using every response row and every Sanger model in the complete outer-training fold.
7. Predict the untouched outer-test rows and evaluate in original endpoint units.

The same process applies to full_graph, drug_id_only, and expression_only. This allows the final variants and existing baselines to use the same outer-training cell-line/response-row scope. It does not remove all methodological differences between estimator families, but it closes the specific training-volume difference documented in the previous audit.

## Leakage checks

- Fit and validation Sanger groups are disjoint.
- Their union must equal the complete outer-training Sanger groups.
- The outer-training and outer-test Sanger groups must be disjoint.
- Fit and validation curve IDs must be disjoint and their union must equal the complete outer-training curve IDs.
- The outer-training and outer-test curve IDs must be disjoint.
- Expression preprocessing uses the complete outer-training cell lines only during final refit.
- Endpoint target means and standard deviations use only the complete outer-training response rows during final refit.
- Outer-test response labels are used only after prediction for metrics and OOF output.
- The final checkpoint stores the refitted model state, full-fold preprocessing values, endpoint scalers, and selected epoch count, while retaining epoch-selection details separately for provenance.

## Tests added

- Final refit partition contains all outer-training cell lines and no outer-test line.
- Final refit preprocessing is unchanged when only held-out expression values and held-out labels are perturbed, as measured on training rows and fitted parameters.
- The end-to-end synthetic refit checkpoint is unchanged when outer-test response labels are perturbed.
- The final refit runs for the epoch count selected on inner validation.
- Missing outer-training response rows are rejected.

## Reproducible command

After local tests pass, run five new seeds with a fresh output directory:

```bash
./.venv/bin/python -m pytest tests/test_gdsc2_pair_gnn_ablations.py -q

./.venv/bin/python scripts/ai/run_gdsc2_pair_gnn_multiseed.py \
  --fresh-run-set \
  --seeds 42 43 44 45 46 \
  --output-root data/ai/gdsc2_pair_gnn_full_outer_refit_3layer
```

The run includes full_graph, drug_id_only, and expression_only variants, the frozen five outer folds, OOF checks, and cluster-bootstrap uncertainty output. Keep these results separate from earlier models, because the earlier metrics were generated before the full-fold refit protocol.

## Status and limitations

The code and regression tests have been added, but local test execution and the fresh real-data five-seed run are still required. This remains a known-drug, held-out-cell-line, transductive GDSC2 in-vitro benchmark; it does not establish unseen-drug performance, patient outcomes, or clinical efficacy.


## Five-seed full outer-training refit results

The five fresh runs completed for seeds 42–46, with 133,929 OOF response records per seed and the same frozen five-fold Sanger-ID split. Each variant used inner validation only to choose epoch count, then a fresh model and complete outer-training fold for final refitting. The uncertainty analysis used 10,000 Sanger-model-cluster bootstrap replicates per seed and comparison.

Mean MAE across five seeds (lower is better):

| Endpoint | Full graph | Drug-ID only | Expression only |
|---|---:|---:|---:|
| LN(IC50) | 1.027829 | 1.023431 | 2.012953 |
| AUC | 0.052937 | 0.053701 | 0.092753 |

Across-seed paired differences in MAE (full graph minus comparator; negative favours full graph):

| Endpoint | Comparator | Mean difference | SD across seeds | Graph lower MAE |
|---|---|---:|---:|---:|
| LN(IC50) | Drug-ID only | +0.004398 | 0.007743 | 1/5 |
| AUC | Drug-ID only | -0.000764 | 0.000577 | 4/5 |
| LN(IC50) | Drug+cancer median | +0.019509 | 0.006709 | 0/5 |
| LN(IC50) | Expression Ridge | +0.041817 | 0.006709 | 0/5 |
| AUC | Drug+cancer median | +0.002899 | 0.000311 | 0/5 |
| AUC | Expression Ridge | +0.000955 | 0.000311 | 0/5 |

The full graph is slightly worse than drug-ID-only on LN(IC50), with a positive mean difference and better full-graph MAE in only one seed. On AUC the full graph is modestly better than drug-ID-only in four of five seeds; seed 43 is very slightly worse (paired MAE difference +0.000093). This is a small gain, not a large separation.

The full graph has higher mean MAE than both external baselines on both endpoints across all five seeds. Reference external baseline MAEs on the same OOF records are: drug+cancer median 1.008320 (LN(IC50)) and 0.050038 (AUC); expression Ridge 0.986012 (LN(IC50)) and 0.051982 (AUC). The paired full-graph-minus-baseline deltas and their cluster-bootstrap intervals are reported in `gdsc2_pair_gnn_uncertainty_summary.json`.

Compared with the earlier three-layer inner-fit-only experiment (mean full-graph MAE 1.02665 LN(IC50), 0.053112 AUC), the full-fold-refit result is nearly unchanged on LN(IC50) (+0.00118) and slightly improved on AUC (-0.000175). This is a descriptive comparison across separately trained experiments, not an architecture significance test.

## Interpretation

The training-volume fairness mismatch is addressed by this run protocol, and the graph-specific benefit over the drug-ID-only control remains limited: absent for LN(IC50) and small for AUC. The full graph remains behind both external baselines on both endpoints. These results do not justify selecting the graph model as the best response predictor.

## Validation status

The real-data five-seed full-fold-refit experiment and uncertainty analysis completed. The local regression-test output for this new branch has not yet been provided, so test validation remains pending here. Review the saved run manifest and fold metrics to ensure every final checkpoint reports `final_refit_used_all_outer_training_rows=true` and refit scalers fitted on the complete outer-training fold.
