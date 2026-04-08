#!/usr/bin/env bash
# Build the wings_engine_patch wheel.
# Usage: build.sh [OUTDIR]
#   OUTDIR  Directory to write the .whl into (default: <repo-root>/build/tmp)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTDIR="${1:-${ROOT_DIR}/build/tmp}"

mkdir -p "${OUTDIR}"
cd "${SCRIPT_DIR}"

echo "[wings-accel] Installing build dependencies..."
if ! python3 -m build --version >/dev/null 2>&1; then
    pip3 install --user build
fi

echo "[wings-accel] Building wings_engine_patch wheel → ${OUTDIR}"
python3 build_wheel.py --outdir "${OUTDIR}"
echo "[wings-accel] ✅ Wheel built in ${OUTDIR}"
