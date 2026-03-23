#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/build/output"
BUILD_DIR="${ROOT_DIR}/build/tmp"

trap 'rm -rf "${BUILD_DIR}"' EXIT

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${BUILD_DIR}"

cd "${ROOT_DIR}/wings_engine_patch"

echo "[wings-accel] Installing build dependencies..."
if ! python3 -m build --version >/dev/null 2>&1; then
    pip3 install --user build
fi

echo "[wings-accel] Building wheel..."
python3 build_wheel.py --outdir "${BUILD_DIR}"

cp "${ROOT_DIR}/install.py" "${BUILD_DIR}/install.py"
cp "${ROOT_DIR}/supported_features.json" "${BUILD_DIR}/supported_features.json"
cp "${BUILD_DIR}"/*.whl "${OUTPUT_DIR}/"
cp "${ROOT_DIR}/install.py" "${OUTPUT_DIR}/install.py"
cp "${ROOT_DIR}/supported_features.json" "${OUTPUT_DIR}/supported_features.json"

tar zcvf "${OUTPUT_DIR}/wings-accel-package.tar.gz" -C "${BUILD_DIR}" . > /dev/null

echo "[wings-accel] ✅ Package: ${OUTPUT_DIR}/wings-accel-package.tar.gz"
