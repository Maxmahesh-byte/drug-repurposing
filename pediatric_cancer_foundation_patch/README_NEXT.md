# First real pediatric-cancer milestone

This patch replaces the current placeholder ingestion/training foundation with a real OpenPedCan metadata pipeline.

## Why this first

The current repository's GNN uses random node features and random training targets. That cannot establish a meaningful scientific benchmark.

This milestone instead gives us:

1. versioned OpenPedCan input
2. local SHA-256 provenance
3. standardized pediatric sample metadata
4. a PyTorch Geometric heterogeneous graph
5. deterministic node mappings

## MacBook setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Download the two core OpenPedCan tables:

```bash
bash scripts/download_openpedcan_core.sh
```

Build the standardized table and graph:

```bash
python -m src.ingestion.data_ingest
python -m src.preprocessing.build_graph
```

Run the test:

```bash
python -m pytest tests/test_openpedcan.py
```

Then inspect:

```bash
ls -lh data/raw/openpedcan/v15
ls -lh data/processed/pediatric_samples.tsv
ls -lh data/processed/pediatric_kg.pt
cat data/metadata/openpedcan_manifest.json
```

## Deliberate next step

Do not train the existing `train_gnn.py` yet. Its current target is random, so its loss is not a scientific result.

Next we should add a real, explicitly defined learning task on the cancer-biology graph, with patient/sample-aware splits and provenance.
