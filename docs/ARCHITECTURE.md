# Software architecture overview

This document describes the repository's software boundaries at a high level. It intentionally does not make claims about scientific validity or the correctness of any domain-specific result.

## Main components

```text
Repository
├── api/                 HTTP API application and request/response schemas
├── frontend/            Streamlit user interface and visual components
├── src/
│   ├── ingestion/       Input-data loading and standardisation
│   ├── preprocessing/   Transformation and graph construction
│   ├── training/        Model lifecycle code
│   ├── inference/       Runtime inference interfaces
│   └── utils/           Shared configuration and utilities
├── data/
│   ├── raw/             Source inputs (local data may be external to Git)
│   └── processed/       Generated intermediate or processed outputs
├── tests/               Automated test modules
├── scripts/             Maintenance, validation, and audit utilities
├── reports/             Generated or curated reports
├── artifacts/           Runtime outputs (normally not source-controlled)
├── .github/workflows/   GitHub Actions automation
├── Dockerfile
├── docker-compose.yml
└── dvc.yaml
```

The diagram is a directory-level map, not a verified runtime call graph.

## Separation of responsibilities

### API

The API boundary is located in `api/`. Keep HTTP concerns, request validation, response schemas, and application orchestration distinct from lower-level processing code.

### Frontend

The user interface is located in `frontend/`. Keep presentation state and visualisation code separate from data access and transformation logic where practical.

### Source packages

The `src/` directory separates input handling, preprocessing, training, inference, and shared utilities. Prefer small modules with explicit inputs and outputs over hidden filesystem or global-state dependencies.

### Data and generated outputs

Keep original inputs, intermediate files, processed outputs, and runtime artifacts distinguishable. Record the source and version of inputs and the transformation that produced each generated file. Do not commit credentials, private data, or large generated artifacts unless the repository's data-governance policy explicitly permits it.

### Tests

The `tests/` directory contains automated test modules. Test existence does not establish that tests pass; the test runner and results must be recorded separately.

## Configuration boundaries

- Keep environment-specific settings out of source code.
- Provide safe example configuration with placeholders only.
- Validate required configuration at application startup.
- Avoid logging credentials or sensitive values.
- Document which service owns each persistent directory or volume.

## Architecture changes

When adding or moving a component:

1. Update this document and the root README.
2. Identify the owning package and its public interface.
3. Add or update tests for the software contract.
4. Update container and pipeline configuration when applicable.
5. Record migration or compatibility notes.
