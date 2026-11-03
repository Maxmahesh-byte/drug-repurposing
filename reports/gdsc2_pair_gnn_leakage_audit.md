# GDSC2 pair-GNN leakage and evaluation audit

## Scope

This audit covers the local GDSC2 in-vitro single-agent response experiment using the exact-name E-MTAB-2770 expression crosswalk. The supervised table contains matched drug/cell-line response pairs; the typed graph contributes drug–ChEMBL–target–gene evidence. It is a known-drug, held-out-cell-line benchmark, not an unseen-drug or patient-outcome evaluation.

## Safeguards implemented

| Audit area | Code-level safeguard | Regression coverage |
|---|---|---|
| Response-edge exclusion | The compact graph builder whitelists only drug→ChEMBL molecule, ChEMBL molecule→direct mechanism target, and target→gene relations. It does not copy the GDSC2 response relation or its edge attributes. | A fixture includes a response edge with an obvious label value and verifies it is absent from the compact graph. |
| Complete gene-to-drug path | The full graph encoder uses three RGCN layers, allowing messages to travel over the reverse path gene→target→ChEMBL molecule→drug. | A gradient test checks whether a gene-node embedding can influence the corresponding drug embedding. |
| Typed graph integrity | Node IDs are unique within each relevant type; each evidence relation is checked for source/target index bounds before conversion. | Tests include an out-of-range edge case. |
| Expression-row identity | Each pair's expression_row_index must equal the NPZ row for its Sanger model ID; graph_node_id must equal GDSC_MODEL:<sanger_model_id>. | Tests deliberately corrupt the expression row and graph node ID and expect failure. |
| Outer/inner split isolation | Sanger model IDs must belong to exactly one frozen outer fold. Inner fit, inner validation, and outer test groups are explicitly checked for pairwise disjointness and full coverage. | Tests cover disjoint groups and reject a Sanger ID assigned to multiple outer folds. |
| Expression preprocessing | Gene variance selection, centering, and scaling use only inner-fit cell lines. The fitted scaler's mean and scale, selected gene indices, and fit scope are saved. | A test perturbs only held-out expression rows and verifies that selected genes, scaler parameters, and transformed training rows do not change. |
| Endpoint scaling | LN(IC50) and AUC have separate mean/standard-deviation scalers fitted from inner-fit response rows. Predictions are inverse-transformed before external-unit metrics. | Unit tests verify endpoint-specific scaler values and inverse-transform behaviour. |
| Held-out labels | Test-row drug/cell indices are built without converting test response values into training tensors. Test outcomes are used after prediction for evaluation only. | The training path makes the label separation explicit; response-edge exclusion is tested independently. |
| Baseline identity | External OOF comparisons require exact curve ID joins and check Sanger model IDs, fold assignments, and measured target values. | Runtime validation fails closed when identifiers, folds, or targets disagree. |
| Evidence coverage | The run summary counts drugs with a ChEMBL mapping, a direct target path, and a complete gene-level path. | A synthetic graph test checks the path counts. |

## Important caveat: training-set size differs from baseline models

The GNN uses an inner-fit subset to update model weights and an inner-validation subset for early stopping. The existing Ridge and drug+cancer median OOF baselines were trained/estimated on the full outer-training fold. They are evaluated on the same held-out response rows and frozen folds, but the number of outer-training observations available to the estimators is not identical.

This is not held-out-label leakage, but it is a training-data fairness difference. A stricter follow-up benchmark should select the epoch on the inner split, then refit the chosen architecture on the full outer-training fold (with expression and endpoint scalers refit on that full outer-training data) before predicting the outer test fold.

## Scope limitations

- Drug nodes and ChEMBL evidence are present for the known drug set during training and testing. Drug embeddings are transductive; this does not evaluate unseen-drug generalisation.
- Static ChEMBL mapping/mechanism edges are evidence annotations, not proof of target engagement or activity in each cell line.
- A graph path is an evidence-coverage statistic, not a biological validation or response biomarker.
- Results are GDSC2 cell-line measurements. They do not establish patient response, pediatric treatment benefit, or clinical efficacy.
- The five-seed results previously recorded used the two-layer architecture. They must not be attributed to the revised three-layer model.
- New tests and the revised training pipeline still require local execution before the audit can be marked passed.

## Required local validation

Run the pair-GNN and ablation tests, then train a fresh five-seed experiment with the corrected three-layer graph encoder. Keep the outputs separate from the earlier two-layer results. Review evidence coverage in the new metrics JSON and compare all variants against the exact same outer-test records.
