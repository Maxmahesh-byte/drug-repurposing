# Graph V2: safe typed-graph migration

This patch is deliberately non-destructive. It reads the existing `data/processed/openpedcan_graph.pt`, creates a PyTorch Geometric `HeteroData` representation with explicit node types and canonical relation directions, and saves the result under `data/processed/graph_v2/`.

It does **not** download source data, overwrite the existing `.pt` files, train a model, or claim that the result is already a fully evidence-preserving research graph. The legacy builder aggregated some source records and dropped some string metadata before saving the original graph. Those details cannot be reconstructed from the `.pt` file alone; raw CSVs remain the source of truth.

## Install files

Extract this patch into your repository root so the files land at:

- `scripts/migrate_graph_v2.py`
- `scripts/validate_graph_v2.py`
- `GRAPH_V2_README.md`

If `scripts/` does not exist, create it first with `mkdir -p scripts`.

## Run migration

From the repository root:

```bash
./.venv/bin/python scripts/migrate_graph_v2.py
```

Expected outputs:

- `data/processed/graph_v2/typed_graph_v2.pt`
- `data/processed/graph_v2/graph_v2_manifest.json`

The migration validates node categories and relation endpoints. Edges whose saved endpoint types contradict the declared relation schema are preserved under a clearly flagged `legacy_type_mismatch__...` relation name. The migration does not guess their intended types; inspect `graph_v2_manifest.json` before modelling.

## Validate the new graph

```bash
./.venv/bin/python scripts/validate_graph_v2.py
```

The validator checks node IDs, local edge indices, edge attributes, and one-to-one mapping back to original edge IDs. It also reports the count of explicitly preserved endpoint-type mismatches.

## Important semantic notes

- Node stores have explicit types; original IDs are retained inside their respective typed stores.
- The original 32-column `x` matrix is one-hot node-type encoding, not biological measurements. The migration intentionally does not expose it as model features.
- The old five-column edge attribute matrix is preserved as `edge_attr_legacy` in the column order `[variant_count, copy_number_count, fusion_count, ln_ic50, auc]`. These columns have relation-dependent meanings and should not be treated as one universal feature vector.
- Old weights are preserved as `edge_weight_legacy`, not labelled as confidence.
- The manifest records known source paths and aggregation notes. It cannot recreate edge-specific ChEMBL mechanism text, source URLs, individual variant records, or sample-specific copy-number calls that the original graph builder did not save.
- Do not delete or replace the original graph or raw CSV files. This is a typed baseline for the next stage of an evidence-preserving rebuild.
- Only load locally generated `.pt` files you trust; PyTorch deserialization can execute Python pickle payloads.


## Legacy type-mismatch note

The original builder uses `networkx.Graph()` and plain strings for node IDs. In such a graph, the same string reused for different entity categories may refer to the same node, and adding an edge between an existing pair may replace its prior edge attributes. If this happened before the legacy `.pt` file was saved, the original intended node categories or relationship cannot always be recovered unambiguously. Graph V2 preserves such edges under `legacy_type_mismatch__<relation>` triplets and records the observed endpoint-type pairs and counts in the manifest. Treat these as audit items, not validated biological relations. The durable fix is to rebuild a later graph from the raw tables with namespaced entity IDs and independent evidence records.
