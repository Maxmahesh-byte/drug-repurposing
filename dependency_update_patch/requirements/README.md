# Dependency layout

Python baseline: 3.11.

- base.txt: shared scientific/data dependencies
- ml.txt: PyTorch, PyG, MLflow
- api.txt: FastAPI/Uvicorn
- frontend.txt: Streamlit
- chemistry.txt: RDKit
- data.txt: DVC
- dev.txt: complete development/test/lint environment
- ../requirements.txt: complete application environment

Local development:

    python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements/dev.txt

After installation, run:

    python scripts/check_environment.py
    python -m pytest
