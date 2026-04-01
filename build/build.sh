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
#   arctic_inference-*.tar.gz   (source package)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/build/output"
BUILD_DIR="${ROOT_DIR}/build/tmp"
PKG_DIR="${ROOT_DIR}/build/pkg"

# Python version digits used for platform wheel resolution (e.g. 311)
PYTHON_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"

trap 'rm -rf "${BUILD_DIR}" "${PKG_DIR}"' EXIT

rm -rf "${OUTPUT_DIR}" "${PKG_DIR}"
mkdir -p "${OUTPUT_DIR}" "${BUILD_DIR}" "${PKG_DIR}"

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
# 3. Copy arctic-inference source package from local archive
# ---------------------------------------------------------------------------
echo "[wings-accel] ── Copying arctic-inference source package..."
ARCTIC_DIR="${ROOT_DIR}/arctic-inference"
if ls "${ARCTIC_DIR}"/arctic_inference-*.tar.gz "${ARCTIC_DIR}"/arctic-inference-*.tar.gz 2>/dev/null | head -1 | grep -q .; then
    cp "${ARCTIC_DIR}"/arctic*.tar.gz "${PKG_DIR}/"
else
    echo "[wings-accel] Warning: no arctic-inference tarball found in ${ARCTIC_DIR}/"
    echo "[wings-accel]   Run: python3 -c \"import urllib.request,json; resp=urllib.request.urlopen('https://pypi.org/pypi/arctic-inference/json'); d=json.load(resp); url=[u['url'] for u in d['urls'] if u['packagetype']=='sdist'][0]; urllib.request.urlretrieve(url, 'arctic-inference/'+url.split('/')[-1])\""
fi

# ---------------------------------------------------------------------------
# 4. Copy install artifacts
# ---------------------------------------------------------------------------
cp "${ROOT_DIR}/install.py" "${PKG_DIR}/install.py"
cp "${ROOT_DIR}/supported_features.json" "${PKG_DIR}/supported_features.json"

# ---------------------------------------------------------------------------
# 5. Package everything into a flat tar.gz
# ---------------------------------------------------------------------------
echo "[wings-accel] ── Packaging delivery archive..."
tar zcf "${OUTPUT_DIR}/wings-accel-package.tar.gz" -C "${PKG_DIR}" .

echo "[wings-accel] ✅ Package: ${OUTPUT_DIR}/wings-accel-package.tar.gz"
echo "[wings-accel]    Contents:"
tar tf "${OUTPUT_DIR}/wings-accel-package.tar.gz" | sed 's/^/     /'
