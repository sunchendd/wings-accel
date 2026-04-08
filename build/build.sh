#!/usr/bin/env bash
# Main build script for wings-accel delivery package.
#
# Output: build/output/wings-accel-package.tar.gz
#
# The tar.gz extracts to a flat directory containing:
#   install.py
#   supported_features.json
#   wings_engine_patch-*.whl
#   wrapt-*-linux_x86_64.whl
#   wrapt-*-linux_aarch64.whl
#   arctic_inference-*.whl      (pre-built wheel, placed in arctic-inference/ before build)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/build/output"
BUILD_DIR="${ROOT_DIR}/build/tmp"
PKG_DIR="${ROOT_DIR}/build/pkg"

# Python version digits used for platform wheel resolution (e.g. 311)
PYTHON_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"

rm -rf "${OUTPUT_DIR}" "${BUILD_DIR}" "${PKG_DIR}"
mkdir -p "${OUTPUT_DIR}" "${BUILD_DIR}" "${PKG_DIR}"

ARCH=$(uname -m)
if [ "${ARCH}" = "x86_64" ]; then
    echo "[wings-accel] ── Building sparsekv wheel (x86_64)..."
    bash "${ROOT_DIR}/sparsekv/build.sh" "${PKG_DIR}"
else
    echo "[wings-accel] ── Skipping sparsekv wheel (not needed for ${ARCH})"
fi

cd "${ROOT_DIR}/wings_engine_patch"

# ---------------------------------------------------------------------------
# 1. Build wings_engine_patch wheel
# ---------------------------------------------------------------------------
echo "[wings-accel] ── Building wings_engine_patch wheel..."
bash "${ROOT_DIR}/wings_engine_patch/build.sh" "${BUILD_DIR}"

# Copy wheel to staging directory
cp "${BUILD_DIR}"/wings_engine_patch-*.whl "${PKG_DIR}/"

# ---------------------------------------------------------------------------
# 2. Download wrapt wheels (x86_64 + aarch64) for offline installation
# ---------------------------------------------------------------------------
echo "[wings-accel] ── Downloading wrapt wheels (x86_64 + aarch64)..."
for PLATFORM in linux_x86_64 linux_aarch64; do
    pip3 download wrapt \
        --platform "${PLATFORM}" \
        --python-version "${PYTHON_VER}" \
        --implementation cp \
        --abi "cp${PYTHON_VER}" \
        --only-binary=:all: \
        -d "${PKG_DIR}" \
        --no-deps \
        --quiet \
        || echo "[wings-accel] Warning: could not download wrapt for ${PLATFORM} (cp${PYTHON_VER})"
done

# ---------------------------------------------------------------------------
# 3. Build arctic-inference wheel (x86_64 only)
# ---------------------------------------------------------------------------
if [ "${ARCH}" = "x86_64" ]; then
    echo "[wings-accel] ── Building arctic-inference wheel (x86_64)..."
    bash "${ROOT_DIR}/arctic-inference/build.sh" "${PKG_DIR}"
else
    echo "[wings-accel] ── Skipping arctic-inference wheel (not needed for ${ARCH})"
fi

cp "${BUILD_DIR}"/*.whl "${PKG_DIR}/"
cp "${ROOT_DIR}/install.py" "${PKG_DIR}/install.py"
cp "${ROOT_DIR}/supported_features.json" "${PKG_DIR}/supported_features.json"

# ---------------------------------------------------------------------------
# 4. Package everything into a flat tar.gz
# ---------------------------------------------------------------------------
echo "[wings-accel] ── Packaging delivery archive..."
tar zcf "${OUTPUT_DIR}/wings-accel-package.tar.gz" -C "${PKG_DIR}" .

echo "[wings-accel] ✅ Package: ${OUTPUT_DIR}/wings-accel-package.tar.gz"
echo "[wings-accel]    Contents:"
tar tf "${OUTPUT_DIR}/wings-accel-package.tar.gz" | sed 's/^/     /'
