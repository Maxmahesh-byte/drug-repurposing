# Evidence Atlas — Research UI and Evidence Index Patch

This patch replaces the simple Streamlit page with a multi-workspace, evidence-first research interface for the existing paediatric oncology project.

## What is included

- `frontend/app.py` — redesigned Streamlit application with six workspaces:
  1. Command centre
  2. Drug–disease investigation
  3. Evidence relationship map
  4. Sample explorer
  5. Provenance & quality
  6. Investor briefing
- `scripts/build_research_evidence_index.py` — one-time, read-only-source index builder that creates a separate SQLite database from existing OpenPedCan, GDSC2 and ChEMBL CSVs.
- `scripts/validate_research_evidence_index.py` — read-only structural validation of that index.
- `requirements-dashboard.txt` — UI dependencies including Plotly.

## Important scientific boundaries

The UI intentionally does **not** generate a drug efficacy score or clinical recommendation.

- ChEMBL records are displayed as source target/mechanism annotations. They do not prove target engagement in a paediatric tumour.
- Exact overlap between ChEMBL target gene symbols and OpenPedCan gene-associated variant observations is a descriptive observation. It does not establish dependency, pathogenicity, pathway activation, or therapeutic fit.
- Intergenic `IGR`/`Unknown` records are excluded from gene-overlap summaries. They remain available in the separate intergenic evidence index and sample explorer. No nearest-gene or regulatory relationship is invented.
- GDSC2 values are in-vitro cell-line response measurements, not patient outcomes. The UI keeps GDSC cancer labels separate from OpenPedCan tumour labels; there is no automatic disease-label equivalence.
- Mutation summary counts are source-record observations, not unique patients.
- The relationship map draws only rows from the local ChEMBL extract and exact gene-symbol observation overlap. Dashed edges mean overlap only, not causality.
- “Why this disease?” is therefore an evidence-exploration workflow, not a validated disease rationale engine yet.
- “Does it look like it will work?” currently displays descriptive in-vitro measurements only. A predictive model requires a defined task, leakage-resistant evaluation, independent validation and appropriate paediatric evidence.

## Expected existing source files

The index builder expects these files, already present in the project inventory shared during development:

```text
data/raw/openpedcan/openpedcan_phase1.csv
data/raw/openpedcan/openpedcan_mutations_phase2.csv
data/raw/gdsc/gdsc2_responses.csv
data/raw/chembl/gdsc2_chembl_target_evidence.csv
```

It creates:

```text
data/processed/graph_v3/research_evidence.sqlite
data/processed/graph_v3/research_evidence_index_manifest.json
```

The new index is separate from the existing intergenic SQLite database, Graph V2 and Graph V3. The source CSVs and graph artifacts are never overwritten.

## Installation and application

Run commands from the repository root.

### 1. Install the patch

Unzip this package into the repository root so the folder `evidence_atlas_ui_patch` is visible. From the repository root, run:

```bash
bash evidence_atlas_ui_patch/install_patch.sh
```

The installer creates timestamped backups of any files it replaces. It changes only `frontend/app.py` and adds the two research-index scripts plus a dashboard requirements file. It does not modify the API, training, inference, raw data, Graph V2, Graph V3 or the existing intergenic index.

### 2. Install UI dependencies

```bash
./.venv/bin/pip install -r requirements-dashboard.txt
```

### 3. Build the research evidence index once

This streams the 18-million-row mutation CSV and imports the GDSC2 response and ChEMBL evidence rows. It may take several minutes. It does not download source data.

```bash
./.venv/bin/python -m py_compile scripts/build_research_evidence_index.py
./.venv/bin/python -u scripts/build_research_evidence_index.py
```

The builder refuses to overwrite an existing research index. If a rebuild is required after source files change, first back up and deliberately remove both:

```text
data/processed/graph_v3/research_evidence.sqlite
data/processed/graph_v3/research_evidence_index_manifest.json
```

Do not remove the existing `intergenic_evidence.sqlite` or Graph V3 files.

### 4. Validate the index

```bash
./.venv/bin/python scripts/validate_research_evidence_index.py
```

Expected final line: `RESULT: PASS`.

### 5. Run the dashboard

```bash
./.venv/bin/python -m streamlit run frontend/app.py
```

## What each workspace demonstrates

### Command centre

Shows independent dataset counts, tumour-type biospecimen distribution and current evidence-domain coverage. Counts have different denominators and should not be interpreted as one shared cohort.

### Drug–disease investigation

Three evidence questions:

1. **What does it hit?** Exact drug-to-target annotations in the current ChEMBL extract, including matching method and source IDs.
2. **Why this disease?** Exact overlap of ChEMBL gene symbols with gene-associated OpenPedCan variant observations for the selected tumour label. This is exploratory and non-causal.
3. **Does it look like it will work?** GDSC2 in-vitro response distributions. The GDSC cancer label is selected separately; the application does not silently equate it with the OpenPedCan tumour label.

### Evidence relationship map

A layered view of source-backed ChEMBL molecule/target/gene links and, where present, exact gene-symbol observation overlap. GDSC response data is displayed as a separate evidence lane.

### Sample explorer

Retains the intergenic evidence lookup using `data/processed/graph_v3/intergenic_evidence.sqlite`, including source row, GRCh38 coordinates, alleles and pagination.

### Provenance & quality

Displays source asset presence, graph manifest checks and the research-index source checksums and data policies.

### Investor briefing

Explains the differentiation thesis, demonstrated engineering milestones and the scientific capabilities that still need validation.

## No changes to prediction or API

This patch changes only the Streamlit frontend and adds a separate research evidence index. It does not modify:

- `src/training/train_gnn.py`
- `src/inference/predict.py`
- API schemas or routes
- Graph V2 or Graph V3 artifacts
- Existing raw CSVs or the intergenic evidence database

Drug-repurposing predictions remain unavailable until a defensible response task and appropriate independent validation exist.
