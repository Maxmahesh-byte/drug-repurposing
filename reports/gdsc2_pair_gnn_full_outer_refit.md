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
