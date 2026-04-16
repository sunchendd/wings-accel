#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

require_cmd git
parse_platform_args "$@"
prepare_build_dirs

if [[ ! -d "${GENERATED_SRC_DIR}" ]]; then
    log "Generated workspace missing; materializing from upstream source first"
    "${SCRIPT_DIR}/materialize_workspace.sh" --platform "${PLATFORM}"
fi

STAMP_FILE="$(patchset_stamp_file)"
PATCHSET_VERSION="$(read_lock_value "patchset_version")"
FORCE_APPLY="false"
if [[ "${#PARSED_ARGS[@]}" -gt 0 ]] && [[ "${PARSED_ARGS[0]}" == "--force" ]]; then
    FORCE_APPLY="true"
fi

if [[ -f "${STAMP_FILE}" ]] && [[ "${FORCE_APPLY}" != "true" ]]; then
    log "Patchset already applied ($(cat "${STAMP_FILE}")); skipping"
    exit 0
fi

PATCH_FILES=()
while IFS= read -r layer; do
    PATCH_DIR="$(patch_dir_for_locked_version "${layer}")"
    if [[ ! -d "${PATCH_DIR}" ]]; then
        if [[ "${layer}" == "common" ]]; then
            fail "Missing common patch directory: ${PATCH_DIR}"
        fi
        continue
    fi
    shopt -s nullglob
    LAYER_PATCHES=("${PATCH_DIR}"/*.patch)
    shopt -u nullglob
    PATCH_FILES+=("${LAYER_PATCHES[@]}")
done < <(patch_layers_for_platform)

if [[ "${#PATCH_FILES[@]}" -eq 0 ]]; then
    warn "No patch files found for platform=${PLATFORM}"
fi

for patch_file in "${PATCH_FILES[@]}"; do
    if placeholder_patch "${patch_file}"; then
        warn "Skipping placeholder patch $(basename "${patch_file}")"
        continue
    fi

    log "Applying patch $(basename "${patch_file}")"
    git -C "${GENERATED_SRC_DIR}" apply --whitespace=nowarn "${patch_file}"
done

echo "${PATCHSET_VERSION}" > "${STAMP_FILE}"
log "Patchset application complete for platform=${PLATFORM}"
