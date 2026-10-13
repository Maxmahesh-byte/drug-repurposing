#!/usr/bin/env bash
set -euo pipefail

RELEASE="${OPENPEDCAN_RELEASE:-v15}"
BASE_URL="${OPENPEDCAN_BASE_URL:-https://s3.amazonaws.com/d3b-openaccess-us-east-1-prd-pbta/open-targets}"
OUT_DIR="data/raw/openpedcan/${RELEASE}"

mkdir -p "${OUT_DIR}"

for FILE in histologies.tsv efo-mondo-map.tsv; do
  echo "Downloading ${FILE}"
  curl -L --fail --retry 3 \
    "${BASE_URL}/${RELEASE}/${FILE}" \
    -o "${OUT_DIR}/${FILE}"
done

echo
echo "OpenPedCan core files are in ${OUT_DIR}"
echo "Next: python src/ingestion/data_ingest.py"
