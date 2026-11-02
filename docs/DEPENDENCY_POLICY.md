# Dependency management policy

## Current repository observation

The root `requirements.txt` pins package versions. A separate `requirements-dashboard.txt` also exists. No root `pyproject.toml` or conventional dependency lockfile was present in the reviewed root listing.

This is an inventory observation, not a recommendation to upgrade any package.

## Policy

1. Keep dependency declarations in a clearly documented, maintained location.
2. Avoid declaring the same dependency with conflicting versions across files.
3. Record the Python version supported by the project.
4. Separate runtime, development, and optional dependencies where useful.
5. Review licence, maintenance status, security advisories, and compatibility before adopting a dependency.
6. Prefer deliberate, reviewed updates over unbounded upgrades.
7. Record the reason for a dependency change and the checks performed afterward.
8. Do not commit local virtual environments or downloaded package caches.

## Dependency update checklist

- [ ] Identify the component that needs the dependency.
- [ ] Check whether an existing dependency already provides the capability.
- [ ] Check compatibility with the documented Python version.
- [ ] Check the package licence and maintenance status.
- [ ] Pin or lock the resolved version according to the chosen project policy.
- [ ] Run relevant software tests.
- [ ] Update setup documentation.
- [ ] Record the change in the changelog or pull request.
