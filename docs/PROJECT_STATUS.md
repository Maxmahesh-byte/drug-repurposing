# Repository status

**Reviewed revision:** `main` at `85eb0f7826d0bf07cfde88786a1cc6ceba21f39d`  
**Review date:** 2026-11-02  
**Review scope:** Repository structure, documentation, configuration, and visible automation metadata.

## Purpose of this document

This is a snapshot of what was visible in the repository at the reviewed revision. It is not a full source-code audit, scientific review, security assessment, or statement that the application has been executed successfully.

## Repository components observed

| Area | Observed contents | Status of this review |
|---|---|---|
| API | `api/main.py`, `api/schemas.py` | File inventory only |
| Frontend | Streamlit application and supporting visualisation module | File inventory only |
| Application source | `src/` with inference, ingestion, preprocessing, training, and utility packages | File inventory only |
| Tests | 14 files under `tests/` | Test files inventoried; tests not executed |
| Data workflow | `dvc.yaml` | Configuration inspected |
| Containers | `Dockerfile`, `docker-compose.yml`, `.devcontainer/` | Configuration inspected |
| Automation | `.github/workflows/retrain.yml` | Workflow definition inspected |
| Reports | Six files under `reports/` | File inventory only |
| Documentation | `README.md`, `GRAPH_V2_README.md`, patch-specific documentation | Read selectively |

## Findings to track

1. The repository has separate API, frontend, source, tests, data, and reporting areas.
2. The root dependency file pins package versions. No root `pyproject.toml` or conventional lockfile was present in the inspected root listing.
3. The DVC configuration declares ingestion and graph-construction stages.
4. Docker Compose declares a pipeline-initialisation service as well as API and frontend services.
5. The README describes limitations on model use, while the Compose initialisation command invokes a training module. This documentation/configuration relationship should be reviewed and clarified by the maintainer.
6. The inspected GitHub Actions workflow is a scheduled/manual workflow. The repository inventory did not show a separate general-purpose test workflow.
7. Several timestamped backup files and patch directories are present. Their status (active, archival, or awaiting integration) should be documented.
8. A dedicated top-level `docs/` directory did not exist at the reviewed revision; this document begins that documentation area.

## Verification status

The following have **not** been verified by this review:

- A successful clean installation.
- A successful Docker Compose startup.
- Successful execution of all DVC stages.
- A passing test suite on the reviewed revision.
- Reproducibility from a clean environment.
- Runtime behaviour of the API or frontend.
- Security, privacy, or licensing compliance.

## Next maintenance actions

- Keep this status page aligned with the default branch.
- Add or update architecture and developer documentation when components change.
- Record actual command output and CI links when checks are run.
- Mark features as implemented only when their behaviour is verified.
