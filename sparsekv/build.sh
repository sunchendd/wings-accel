#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTDIR="${1:-${ROOT_DIR}/build/output}"

DOCKER_IMAGE="docker.artifactrepo.wux-g.tools.xfusion.com/ai_solution/ci/wings/x86/vllm-openai_cmake_3.30.3:v0.17.0"
CONTAINER_NAME="wings-sparsekv-builder-$$"

cleanup() {
	docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}

trap cleanup EXIT

if [[ "$(uname -m)" != "x86_64" ]]; then
	echo "[wings-accel] Error: sparsekv container build currently supports x86_64 only."
	exit 1
fi

mkdir -p "${OUTDIR}"

echo "[wings-accel] ── Building sparsekv wheel in Docker..."
docker run --name "${CONTAINER_NAME}" -d "${DOCKER_IMAGE}" sleep 3600 >/dev/null
docker cp "${SCRIPT_DIR}" "${CONTAINER_NAME}:/tmp/sparsekv"

docker exec "${CONTAINER_NAME}" /bin/bash -lc '
set -euo pipefail
cd /tmp/sparsekv

rm -rf build build_* dist native/*.so third_party/cutlass third_party/cutlass-main third_party/cutlass-main.zip
mkdir -p third_party

if ! command -v unzip >/dev/null 2>&1; then
	apt-get update >/dev/null
	apt-get install -y unzip >/dev/null
fi

if ! command -v cmake >/dev/null 2>&1; then
	apt-get update >/dev/null
	apt-get install -y cmake >/dev/null
fi

wget -q -O third_party/cutlass-main.zip https://artifactrepo.wux-g.tools.xfusion.com/artifactory/opensource_general/cutlass/4.3.3/package/cutlass-main.zip --no-check-certificate
unzip -q third_party/cutlass-main.zip -d third_party/
mv third_party/cutlass-main third_party/cutlass

python3 -m pip install --quiet wheel cmake packaging "setuptools<81" pybind11 ninja build
python3 - <<"PY"
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("torch") is None:
	subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "torch"])
PY

MAX_JOBS=$(nproc) python3 setup.py build_ext
python3 -m build --wheel --no-isolation
'

WHEEL_PATH="$(docker exec "${CONTAINER_NAME}" /bin/bash -lc 'ls /tmp/sparsekv/dist/vsparse-*.whl | head -1')"
if [[ -z "${WHEEL_PATH}" ]]; then
	echo "[wings-accel] Error: sparsekv wheel not found in container."
	exit 1
fi

docker cp "${CONTAINER_NAME}:${WHEEL_PATH}" "${OUTDIR}/"
echo "[wings-accel] ✅ SparseKV wheel built: $(basename "${WHEEL_PATH}")"
