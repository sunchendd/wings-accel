#!/usr/bin/env bash
# Build the arctic-inference wheel from the local source tarball.
#
# This script must run on a machine that has all build-time dependencies:
#   cmake >= 3.12, ninja, grpcio-tools, torch == 2.7.0, nanobind == 2.9.2
#
# Usage: build.sh [OUTDIR]
#   OUTDIR  Directory to write the .whl into (default: same directory as this script)
#
# The generated arctic_inference-*.whl is placed in OUTDIR and is then
# consumed by build/build.sh to assemble the delivery package.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${1:-${SCRIPT_DIR}}"

# ---------------------------------------------------------------------------
# Locate source tarball
# ---------------------------------------------------------------------------
SRC="$(find "${SCRIPT_DIR}" -maxdepth 1 \
    \( -name "arctic_inference-*.tar.gz" -o -name "arctic-inference-*.tar.gz" \) \
    2>/dev/null | sort | tail -1)"

if [ -z "${SRC}" ]; then
    echo "[arctic-inference] Error: no source tarball found in ${SCRIPT_DIR}/"
    echo "[arctic-inference]   Place arctic_inference-*.tar.gz there before building."
    exit 1
fi

echo "[arctic-inference] ── Building wheel from $(basename "${SRC}") ..."
mkdir -p "${OUTDIR}"

# ---------------------------------------------------------------------------
# Build wheel (pip wheel honours pyproject.toml build-system requirements)
# ---------------------------------------------------------------------------
pip3 wheel "${SRC}" \
    --wheel-dir "${OUTDIR}" \
    --no-deps \
    --quiet

WHL="$(find "${OUTDIR}" -maxdepth 1 -name "arctic_inference-*.whl" | sort | tail -1)"
if [ -z "${WHL}" ]; then
    echo "[arctic-inference] Error: wheel not found in ${OUTDIR} after build."
    exit 1
fi

echo "[arctic-inference] ✅ Wheel built: $(basename "${WHL}")"
