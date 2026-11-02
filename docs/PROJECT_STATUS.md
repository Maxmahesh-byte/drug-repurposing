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
2. The root requirements file pins direct dependency versions, while a separate dashboard requirements file uses version ranges. The Dockerfile installs only the root requirements file.
3. The inspected root listing contains no root `pyproject.toml` or conventional fully resolved dependency lockfile.
4. The Dockerfile and inspected workflow specify Python 3.11; the Dev Container configuration does not itself declare a Python version.
5. The DVC configuration declares two stages. The Compose initializer contains an additional operation not represented as a separate DVC stage.
6. The Compose services use short-form `depends_on`. This starts dependencies in order but does not wait for a dependency to complete successfully, despite the Compose comment implying completion ordering.
7. The Dockerfile default command and the service-specific Compose commands use different process layouts.
8. The Docker ignore file excludes `data/`, while Compose bind-mounts host data directories at runtime. It also contains a singular `report/` pattern while the observed repository folder is `reports/`.
9. The inspected GitHub Actions workflow is a scheduled/manual workflow. The repository inventory did not show a separate general-purpose test workflow.
10. Several timestamped backup files and patch directories are present. Their status (active, archival, or awaiting integration) should be documented.
11. A dedicated top-level `docs/` directory did not exist at the reviewed revision; this documentation set begins that area.

See [Configuration consistency audit](CONFIGURATION_AUDIT.md) for details and proposed maintainer decisions.

## Verification status

The following have **not** been verified by this review:

- A successful clean installation.
- A successful Docker image build.
- A successful Docker Compose startup.
- Successful execution of all DVC stages.
- A passing test suite on the reviewed revision.
- Reproducibility from a clean environment.
- Runtime behaviour of the API or frontend.
- Security, privacy, or licensing compliance.

## Next maintenance actions

- Keep this status page aligned with the default branch.
- Resolve or explicitly accept the configuration findings in the audit.
- Record actual command output and CI links when checks are run.
- Mark features as implemented only when their behaviour is verified.
