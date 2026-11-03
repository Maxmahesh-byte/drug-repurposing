# Multimodal drug–cell-line GNN upgrade

## Scope

Implement all seven workstreams as one coordinated upgrade while retaining the current R-GCN experiment as the immutable comparator. The prediction task remains the known-drug, held-out-cell-line GDSC2 in-vitro benchmark for LN(IC50) and AUC. It is not a patient-response or treatment recommendation model.

## Current state (audited 2026-11-03)

- The existing encoder is a relational GNN: PyG `RGCNConv`, not plain `GCNConv`.
- The pair model currently uses drug-side graph embeddings, CCLE expression embeddings, and a pair head based on drug, cell, product, and absolute difference.
- The graph extraction code now has an OpenPedCan context branch in PR #12, but it still uses the legacy typed graph and therefore inherits known identity and evidence aggregation limitations.
- `DrugCellLinePairGNN` has an optional chemical-feature encoder, but the inspected ablation training path constructs the model without `num_chemical_features` and does not pass `chemical_features` to `encode`. The chemistry branch is therefore not active end-to-end.
- Current OpenPedCan copy-number edges are tumour-type aggregated, not sample-level observations. OpenPedCan patient biospecimens must not be treated as GDSC cell lines.
- GDSC response edges and response-valued fields must remain excluded from message passing and all feature construction.

## Seven workstreams

### 1. Evidence-preserving heterogeneous graph

Build the next evidence-preserving typed graph revision directly from source tables, reusing the existing PyG HeteroData design and V2/V3 artifacts. Use namespaced node IDs, stable HGNC/Ensembl gene crosswalks, observation/evidence nodes, source record identifiers, release/version metadata, match state, and explicit missingness. Restore sample-level mutation, copy-number, and fusion observations rather than relying on legacy aggregated edges. Keep all response labels out of the graph encoder.

### 2. Drug identity and chemical structure

Expand GDSC drug mapping using authoritative identifiers and synonym evidence. Preserve exact, synonym, ambiguous, unresolved, salt, and parent-compound states. Retrieve ChEMBL canonical SMILES and InChI/InChIKey with provenance; compute RDKit Morgan fingerprints and molecular descriptors. Add molecular-graph encoding only as a controlled additional experiment. Integrate the chemical matrix through dataset preparation, fold-specific training/refit, validation, prediction, and saved model metadata. Do not infer chemical equivalence from fuzzy name similarity.

### 3. Cell-line-matched omics

Audit available GDSC/CCLE/DepMap/Cell Model Passport mutation, CNV, fusion, and pathway features. Join only through verified stable cell-model identifiers and documented crosswalks. Fit feature selection, imputation, scaling, and dimensionality reduction using training-fold cell lines only. Do not transfer patient-level OpenPedCan measurements onto cell lines.

### 4. OpenPedCan pediatric context and pathways

Normalize OpenPedCan gene identities and preserve sample-level variant/CNV/fusion evidence. Add curated pathway memberships and drug-target/pathway relationships (e.g. Reactome) with stable source IDs and release provenance. Shared gene nodes provide biological context flow; they do not establish patient-to-cell-line identity or response labels. Evaluate pediatric context as a distinct ablation.

### 5. Drug–cell-line interaction model

Move beyond only concatenating drug and cell embeddings. Test target-/pathway-conditioned gating or cross-attention so the cell context can modulate relevant drug-target evidence. Retain the current pair head as the comparator. Guard against target evidence being interpreted as proof of activity in a specific cell line.

### 6. Architecture comparisons

Compare R-GCN, GATv2/heterogeneous attention, and HGT with identical data, features, folds, seeds, optimization budgets, and stopping rules. Architecture must be isolated from feature changes; no claim that one architecture is an industry standard or inherently superior.

### 7. Evaluation and generalization

Keep the current frozen cell-line-grouped five-fold benchmark for comparability. Add drug-held-out evaluation for chemical transfer and, if supported, cold-drug/cold-cell-line evaluation. Report pooled and macro-drug metrics, per-drug/per-cell-line metrics, uncertainty, and cluster-aware intervals. Compare against drug-ID, expression Ridge, and drug+cancer-median baselines. Preserve previous outputs and use new output directories.

## Required staged ablations

1. Reproduce the current baseline.
2. Add chemical features only.
3. Add cell-line-matched mutation/CNV/fusion features.
4. Add evidence-preserving OpenPedCan context.
5. Add curated pathway evidence.
6. Add target-/pathway-conditioned interaction head.
7. Compare R-GCN, GATv2, and HGT under matched settings.
8. Run drug-held-out and any supported cold-start evaluations.

Each stage must have unit tests, data lineage, feature/missingness audits, and a fresh output directory. Do not combine all changes into one opaque comparison.

## Scientific and leakage safeguards

- Never infer sample identity from tumour type or shared gene membership.
- Never use GDSC response labels, response aggregates, or target-derived response annotations as message-passing features.
- No preprocessing may be fit using held-out cell lines or held-out drugs in the relevant evaluation.
- Distinguish evidence existence from functional activity and from predictive association.
- Report failures and negative results; do not promise a score increase.
- A component is accepted only if its incremental effect is reproducible across seeds/splits and improves the relevant baseline without leakage.

## Immediate next implementation task

Start with the evidence/identifier audit and ChEMBL structure feature pipeline. The first code milestone should create a provenance-rich drug structure table and aligned fingerprint/descriptor matrix, add strict ID/order checks, and wire that matrix through every training and inference path. Then validate a chemistry-only ablation before proceeding to the other modalities.
