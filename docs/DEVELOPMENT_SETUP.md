# Development setup

These instructions cover the Python development environment and general software checks. They do not describe how to operate domain-specific analysis or modelling workflows.

## Requirements

- Python 3.11 (the container and automation configuration currently specify this version).
- Git.
- A virtual-environment tool.
- Docker Compose only when working on container configuration.

## Create an isolated environment

From the repository root:

```bash
python3.11 -m venv .venv
```

Activate it:

**macOS / Linux**
```bash
source .venv/bin/activate
```

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

Install the pinned Python dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The repository currently also contains a dashboard-specific requirements file. Before changing dependencies, confirm which dependency set is intended for the component being developed.

## General software checks

Run the test suite from the repository root:

```bash
python -m pytest
```

Record the Python version, dependency installation result, command, exit status, and test summary when reporting a check.

## Environment variables

Do not commit local `.env` files or secrets. If an environment variable is required:

- Document its purpose.
- Provide a non-sensitive example name/value in documentation.
- Validate it at startup.
- Never include real credentials in examples, logs, issues, or test fixtures.

## Reporting a reproducible issue

Include:

- Operating system and Python version.
- Git commit SHA.
- Exact command and working directory.
- Relevant, redacted error output.
- Whether the issue reproduces in a fresh virtual environment.

Do not include private datasets, credentials, personal information, or unredacted environment files in an issue.
