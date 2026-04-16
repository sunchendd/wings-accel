#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

parse_platform_args "$@"
RUN_BUILD_CHECK="false"
for arg in "${PARSED_ARGS[@]}"; do
    if [[ "${arg}" == "--build" ]]; then
        RUN_BUILD_CHECK="true"
    fi
done

if [[ "${PLATFORM}" == "ascend" ]]; then
    ensure_kvcache_ops_source_ready
fi

log "Verifying offline LMCache workflow for platform=${PLATFORM}"
"${SCRIPT_DIR}/unpack_upstream.sh" --platform "${PLATFORM}"
"${SCRIPT_DIR}/materialize_workspace.sh" --platform "${PLATFORM}"
"${SCRIPT_DIR}/apply_patchset.sh" --platform "${PLATFORM}" --force

while IFS= read -r layer; do
    PATCH_DIR="$(patch_dir_for_locked_version "${layer}")"
    [[ -d "${PATCH_DIR}" ]] || continue
    shopt -s nullglob
    PATCH_FILES=("${PATCH_DIR}"/*.patch)
    shopt -u nullglob
    for patch_file in "${PATCH_FILES[@]}"; do
        if placeholder_patch "${patch_file}"; then
            warn "Placeholder patch still present: $(basename "${patch_file}")"
        fi
    done
done < <(patch_layers_for_platform)

if [[ "${RUN_BUILD_CHECK}" == "true" ]]; then
    log "Running wheel build check"
    "${SCRIPT_DIR}/build_wheel.sh" --platform "${PLATFORM}" --skip-prepare
fi

log "Verification completed successfully for platform=${PLATFORM}"
