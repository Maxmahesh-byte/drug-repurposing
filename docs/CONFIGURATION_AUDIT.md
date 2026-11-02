# Configuration consistency audit

**Reviewed revision:** `main` at `85eb0f7826d0bf07cfde88786a1cc6ceba21f39d`  
**Review date:** 2026-11-02  
**Scope:** Dependency declarations, Docker configuration, Compose orchestration, DVC stage inventory, and ignore rules.

This is a static configuration review. No image was built, no container was started, no dependency was installed, and no application or pipeline command was run.

## Findings

### 1. Compose startup ordering does not match its comment

The `fastapi` and `streamlit` services use the short form of `depends_on` for `pipeline-init`. Docker documents that short-form `depends_on` guarantees dependency startup order, but does not wait for a dependency to become healthy or to complete successfully. The comment in the Compose file says the initializer finishes before the other services start, which is stronger than the configured behaviour.

**Impact:** A dependent service may start while the initializer is still running. The configuration does not itself express a successful-completion gate.

**Maintainer decision:** Confirm whether the initializer is required to complete successfully before the dependent services start. If so, review the long-form Compose dependency condition `service_completed_successfully` and test it with the Compose version used by the project. No runtime change is made in this audit.

Reference: [Docker Compose startup order](https://docs.docker.com/compose/how-tos/startup-order/).

### 2. Dependency files have different roles and resolution policies

- The root `requirements.txt` uses exact version pins.
- `requirements-dashboard.txt` uses version ranges and includes a separate dashboard-oriented set.
- The Dockerfile installs `requirements.txt`; it does not install `requirements-dashboard.txt`.
- The inspected root listing contains no conventional fully resolved dependency lockfile.

**Impact:** The Docker image and a local environment created from the dashboard requirements file are not guaranteed to resolve the same versions. Exact pins for direct requirements do not by themselves freeze every transitive dependency.

**Maintainer decision:** Choose and document one of these approaches: a single authoritative dependency set, or clearly separated base and optional sets with a documented installation matrix. Consider a fully resolved lock strategy if repeatable installs are a project requirement. Do not upgrade packages as part of this documentation review.

### 3. Python version is not declared uniformly in every development environment

The Dockerfile and inspected workflow specify Python 3.11. The Dev Container configuration selects a generic Ubuntu base image and does not itself declare a Python version.

**Impact:** A developer using the Dev Container is not guaranteed by that file alone to use the same Python version as the image and workflow.

**Maintainer decision:** Decide whether the Dev Container should explicitly provision the supported Python version or whether setup instructions should require developers to install it themselves.

### 4. Compose orchestration and the DVC stage graph are not identical

The DVC file declares two stages. The Compose initializer contains an additional sequential operation that is not represented as a separate stage in the DVC graph.

**Impact:** The DVC stage graph does not describe every operation performed by the Compose initializer. This can make it harder to understand which workflow is authoritative and what is tracked by DVC.

**Maintainer decision:** Document the intended relationship between the two orchestration mechanisms. Any change to pipeline stages or commands requires review by the project maintainer and is outside this general configuration-only change.

### 5. Dockerfile default command differs from Compose service commands

The Dockerfile default command starts both the API server and the Streamlit interface from one container command. The Compose API and frontend services each provide their own command, overriding the image default.

**Impact:** Running the image directly and running the project through Compose use different process layouts.

**Maintainer decision:** State which deployment mode is supported. If the combined default is not intended for direct image use, consider simplifying it in a separately reviewed change.

### 6. Build context and runtime data mounts are different by design

The Docker ignore file excludes the `data/` directory from the build context. Compose separately bind-mounts the host `./data` directory into the runtime containers.

**Impact:** Local data is not copied into the image during build, but it is exposed to containers at runtime through the bind mount. This separation should be explicit in setup and data-governance documentation.

**Maintainer decision:** Preserve this separation unless there is an explicitly approved data packaging strategy. Do not add private or large datasets to the image by default.

### 7. Reports ignore pattern appears singular while the repository folder is plural

The Docker ignore file contains `report/`, while the repository directory is named `reports/`.

**Impact:** The current pattern does not document an exclusion for the observed plural directory. As a result, report files may enter the Docker build context.

**Maintainer decision:** Confirm whether `reports/` should be part of the image. If not, update the ignore pattern in a separate, reviewed configuration change.

## Proposed order of resolution

1. Confirm the intended Compose startup dependency semantics.
2. Choose the dependency-file and lock strategy.
3. Decide whether the Dev Container should provision Python 3.11.
4. Clarify the relationship between Compose orchestration and DVC stages.
5. Decide whether the Dockerfile default should support a combined process layout.
6. Confirm the policy for reports in the image build context.

## Changes made in this audit

Documentation only. No application code, dependency versions, container commands, orchestration stages, data files, or model-related behaviour were changed.
