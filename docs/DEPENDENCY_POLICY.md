# Dependency management policy

## Current repository observation

The root `requirements.txt` pins direct package versions. A separate `requirements-dashboard.txt` exists and uses version ranges. The Dockerfile installs the root requirements file only. No root `pyproject.toml` or conventional fully resolved dependency lockfile was present in the reviewed root listing.

Exact pins for direct dependencies do not necessarily pin every transitive dependency resolved by the installer. The two requirements files also do not guarantee the same environment.

This is an inventory observation, not a recommendation to upgrade any package. See [Configuration consistency audit](CONFIGURATION_AUDIT.md) for the detailed findings.

## Policy

1. Keep dependency declarations in a clearly documented, maintained location.
2. Avoid declaring the same dependency with conflicting versions across files.
3. Record the Python version supported by the project.
4. Separate runtime, development, and optional dependencies where useful.
5. Document which dependency file is used by each supported installation route.
6. Review licence, maintenance status, security advisories, and compatibility before adopting a dependency.
7. Prefer deliberate, reviewed updates over unbounded upgrades.
8. Record the reason for a dependency change and the checks performed afterward.
9. Do not commit local virtual environments or downloaded package caches.

## Dependency update checklist

- [ ] Identify the component that needs the dependency.
- [ ] Check whether an existing dependency already provides the capability.
- [ ] Check compatibility with the documented Python version.
- [ ] Check the package licence and maintenance status.
- [ ] Pin or lock the resolved version according to the chosen project policy.
- [ ] Confirm the dependency file used by local setup, CI, and container builds.
- [ ] Run relevant software tests.
- [ ] Update setup documentation.
- [ ] Record the change in the changelog or pull request.
