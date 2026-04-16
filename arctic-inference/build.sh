#!/usr/bin/env bash
# Build the arctic-inference wheel from remote source using Docker.
#
# This script runs the build inside a Docker container to ensure consistent
# build environment with all required dependencies.
#
# Usage: build.sh [OUTDIR]
#   OUTDIR  Directory to write the .whl into (default: same directory as this script)
#
# The generated arctic_inference-*.whl is placed in OUTDIR and is then
# consumed by build/build.sh to assemble the delivery package.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${1:-${SCRIPT_DIR}}"

DOCKER_IMAGE="${WINGS_ARCTIC_DOCKER_IMAGE:-vllm/vllm-openai:v0.17.0}"
CONTAINER_NAME="arctic-inference-builder"

echo "[arctic-inference] ── Building wheel from remote source..."
mkdir -p "${OUTDIR}"

# ---------------------------------------------------------------------------
# Clean up existing container if any
# ---------------------------------------------------------------------------
docker rm -f ${CONTAINER_NAME} 2>/dev/null || true

# ---------------------------------------------------------------------------
# Build wheel in Docker container
# ---------------------------------------------------------------------------
docker run --name ${CONTAINER_NAME} -d ${DOCKER_IMAGE} sleep 3600

echo "[arctic-inference] Downloading and building wheel in container..."
docker exec ${CONTAINER_NAME} /bin/bash -lc '
set -euo pipefail
python3 -m pip wheel arctic-inference==0.1.1 --wheel-dir /tmp --no-deps --quiet \
  || python3 -m pip wheel arctic_inference==0.1.1 --wheel-dir /tmp --no-deps --quiet
'

echo "[arctic-inference] Copying wheel back from container..."
WHL_NAME=$(docker exec ${CONTAINER_NAME} bash -c "ls /tmp/arctic_inference-*.whl 2>/dev/null | head -1" | xargs basename 2>/dev/null)
if [ -n "${WHL_NAME}" ]; then
    docker cp ${CONTAINER_NAME}:/tmp/${WHL_NAME} "${OUTDIR}/"
    echo "[arctic-inference] ✅ Wheel built: ${WHL_NAME}"
else
    echo "[arctic-inference] Error: wheel not found in container."
    docker rm -f ${CONTAINER_NAME}
    exit 1
fi

docker rm -f ${CONTAINER_NAME}

WHL="${OUTDIR}/${WHL_NAME}"
if [ -z "${WHL}" ]; then
    echo "[arctic-inference] Error: wheel not found in ${OUTDIR} after build."
    exit 1
fi

echo "[arctic-inference] ✅ Wheel built: $(basename "${WHL}")"
