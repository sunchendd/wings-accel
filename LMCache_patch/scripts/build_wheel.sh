#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

require_cmd python3
parse_platform_args "$@"
prepare_build_dirs

SKIP_PREPARE="false"
if [[ "${#PARSED_ARGS[@]}" -gt 0 ]] && [[ "${PARSED_ARGS[0]}" == "--skip-prepare" ]]; then
    SKIP_PREPARE="true"
fi

if [[ "${SKIP_PREPARE}" != "true" ]]; then
    "${SCRIPT_DIR}/unpack_upstream.sh" --platform "${PLATFORM}"
    "${SCRIPT_DIR}/materialize_workspace.sh" --platform "${PLATFORM}"
    "${SCRIPT_DIR}/apply_patchset.sh" --platform "${PLATFORM}" --force
fi

ensure_generated_workspace
stamp_generated_lmcache_version_file

rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}"

LMCACHE_PACKAGE_VERSION="$(locked_python_package_version)"

if [[ "${PLATFORM}" == "common" ]]; then
    log "Attempting optional vendored kv-agent build before LMCache wheel"
    "${SCRIPT_DIR}/build_kv_agent.sh" --platform "${PLATFORM}" --skip-prepare --optional
else
    log "Skipping kv-agent build for platform=${PLATFORM}; Ascend workflow does not use QAT"
fi

pushd "${GENERATED_SRC_DIR}" >/dev/null

if [[ "${PLATFORM}" == "ascend" ]]; then
    export BUILD_WITH_ASCEND="1"
    log "Platform ascend selected; exporting BUILD_WITH_ASCEND=1 for downstream build logic"
fi

export SETUPTOOLS_SCM_PRETEND_VERSION="${LMCACHE_PACKAGE_VERSION}"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE="${LMCACHE_PACKAGE_VERSION}"
log "Forcing LMCache package version to ${LMCACHE_PACKAGE_VERSION}"

if [[ -f "setup.py" ]]; then
    log "Building LMCache wheel with setup.py bdist_wheel to reuse the current environment's installed torch"
    python3 setup.py bdist_wheel --dist-dir "${DIST_DIR}"
elif python3 -m build --help >/dev/null 2>&1; then
    log "setup.py not found; falling back to python -m build"
    python3 -m build --wheel --no-isolation --outdir "${DIST_DIR}"
else
    fail "Unable to build wheel: neither python -m build nor setup.py is available"
fi

popd >/dev/null

log "Wheel build complete for platform=${PLATFORM}. Output directory: ${DIST_DIR}"
find "${DIST_DIR}" -maxdepth 1 -type f -name '*.whl' -print
