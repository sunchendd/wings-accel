#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/build/output"

mkdir -p "${OUTPUT_DIR}"
find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type f -delete

cd "${ROOT_DIR}/wings_engine_patch"

echo "[wings-accel] Installing dev dependencies..."
pip install -q -r "${ROOT_DIR}/requirements-dev.txt"

echo "[wings-accel] Building wheel..."
python3 build_wheel.py --outdir "${OUTPUT_DIR}"

cp "${ROOT_DIR}/install.py" "${OUTPUT_DIR}/install.py"
cp "${ROOT_DIR}/supported_features.json" "${OUTPUT_DIR}/supported_features.json"

echo "[wings-accel] ✅ Deliverables written to ${OUTPUT_DIR}"
