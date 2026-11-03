# GDSC2 nested drug-evidence graph ablations

## Research question

Does progressively adding curated drug-evidence relationships improve held-out GDSC2 drug-response prediction beyond drug identity plus cell-line expression?

## Controlled variants

All variants use the same frozen five-fold GroupKFold split by Sanger model ID, the same inner validation split per seed/fold, the same expression preprocessing, endpoint scaling, training/refit procedure, optimizer configuration, hidden dimension, dropout, pair head, and three-layer R-GCN architecture where applicable.

1. **drug_id_only** — learned drug-ID embedding + cell-line expression; no graph propagation.
2. **chembl_mapping** — adds drug ↔ ChEMBL molecule mapping edges.
3. **target_evidence** — adds ChEMBL molecule ↔ direct mechanism target edges.
4. **gene_evidence** — adds target ↔ gene-symbol annotation edges; this is the complete currently available drug evidence graph.
5. **expression_only** — expression encoder only; retained as a diagnostic lower-level control.

For graph variants, node universe, entity-ID embeddings, node-type embeddings, relation parameter dimensions, and model architecture are held constant. Only the allowed message-passing edge types are progressively enabled. This avoids changing model width or embedding-table size between graph levels.

The three forward relations are assigned relation IDs 0, 1, and 2; their reverse relations use IDs 3, 4, and 5. Thus the allowed relation IDs are:
- ChEMBL mapping: 0 and 3
- Direct target evidence: 0, 1, 3, and 4
- Gene evidence: 0 through 5

## Important terminology limitation

The available graph has ChEMBL molecule identifier nodes and curated mapping/target/gene evidence edges. It does **not** contain atom-and-bond molecular graphs, molecular fingerprints, or structure-derived features in this pipeline. This experiment therefore tests progressively richer **drug knowledge-graph evidence**, not chemical-structure GNN features. A true molecular-structure experiment requires validated SMILES/structure retrieval, salt/parent handling, canonicalisation, and a separate atom/bond graph or fingerprint pipeline.

## Leakage and evaluation safeguards

- GDSC2 response edges and response edge attributes are excluded from message passing.
- Outer test cell lines and response curves are excluded from training, feature selection, expression scaling, and endpoint scaling.
- Each variant selects its epoch using inner validation, then reinitialises and refits on the complete outer-training fold.
- Every variant predicts the same outer-test curves.
- The multi-seed runner checks OOF curve IDs, Sanger model IDs, folds, and measured response values across seeds.
- Paired uncertainty uses Sanger model ID as the resampling cluster, stratified within outer folds.

## Run

First pull the branch and run the focused tests plus the full suite:

```bash
git pull
.venv/bin/python -m pytest tests/test_gdsc2_pair_gnn.py tests/test_gdsc2_pair_gnn_ablations.py -q
.venv/bin/python -m pytest -q
```

Then run five fresh seeds into a new output directory:

```bash
.venv/bin/python scripts/ai/run_gdsc2_pair_gnn_multiseed.py \
  --fresh-run-set \
  --seeds 42 43 44 45 46 \
  --output-root data/ai/gdsc2_evidence_layer_ablations
```

Outputs include per-seed OOF predictions, fold checkpoints and metrics, a multi-seed manifest, and paired cluster-bootstrap comparisons between adjacent evidence levels and key controls.

## Interpretation plan

- Compare ChEMBL mapping against drug-ID-only to test whether the mapping layer helps.
- Compare target evidence against ChEMBL mapping to test whether direct target annotations add value.
- Compare gene evidence against target evidence to test whether gene-symbol annotations add value.
- Compare complete gene evidence against drug-ID-only and expression-only to quantify the total graph contribution.
- Retain external drug+cancer median and expression Ridge comparisons as benchmark references.

A lower MAE is better. Report both endpoints, per-seed results, across-seed variability, and cluster-bootstrap intervals. Do not select a winning graph level based only on one seed or one endpoint.

## Scope

Known-drug, held-out-cell-line, transductive GDSC2 in-vitro benchmark. The experiment does not test unseen-drug generalisation, clinical outcomes, or treatment recommendations.
