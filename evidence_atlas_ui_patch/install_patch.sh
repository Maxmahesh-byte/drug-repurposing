#!/usr/bin/env bash
set -euo pipefail

PATCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"

if [[ ! -f "$PROJECT_ROOT/api/main.py" || ! -d "$PROJECT_ROOT/data" ]]; then
  echo "Run this installer from the drug-repurposing repository root."
  exit 1
fi

mkdir -p "$PROJECT_ROOT/frontend" "$PROJECT_ROOT/scripts"

backup_if_present() {
  local target="$1"
  if [[ -f "$target" ]]; then
    cp -p "$target" "${target}.before_evidence_atlas_${STAMP}"
    echo "Backup: ${target}.before_evidence_atlas_${STAMP}"
  fi
}

backup_if_present "$PROJECT_ROOT/frontend/app.py"
backup_if_present "$PROJECT_ROOT/scripts/build_research_evidence_index.py"
backup_if_present "$PROJECT_ROOT/scripts/validate_research_evidence_index.py"
backup_if_present "$PROJECT_ROOT/requirements-dashboard.txt"

cp "$PATCH_ROOT/frontend/app.py" "$PROJECT_ROOT/frontend/app.py"
cp "$PATCH_ROOT/scripts/build_research_evidence_index.py" "$PROJECT_ROOT/scripts/"
cp "$PATCH_ROOT/scripts/validate_research_evidence_index.py" "$PROJECT_ROOT/scripts/"
cp "$PATCH_ROOT/requirements-dashboard.txt" "$PROJECT_ROOT/requirements-dashboard.txt"

cat <<'EOF'

Evidence Atlas UI patch installed.

Next commands:
  ./.venv/bin/pip install -r requirements-dashboard.txt
  ./.venv/bin/python -m py_compile frontend/app.py scripts/build_research_evidence_index.py scripts/validate_research_evidence_index.py
  ./.venv/bin/python -u scripts/build_research_evidence_index.py
  ./.venv/bin/python scripts/validate_research_evidence_index.py
  ./.venv/bin/python -m streamlit run frontend/app.py

The source CSV files, Graph V2, Graph V3 and existing intergenic SQLite index
were not modified by this installer.
EOF
