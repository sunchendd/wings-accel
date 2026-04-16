#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

parse_platform_args "$@"
prepare_build_dirs

if [[ ! -d "${GENERATED_SRC_DIR}" ]]; then
    log "Generated workspace missing; unpacking upstream source first"
    "${SCRIPT_DIR}/unpack_upstream.sh" --platform "${PLATFORM}"
fi

while IFS= read -r layer; do
    layer_overlay_dir="$(overlay_dir_for_layer "${layer}")"
    [[ -d "${layer_overlay_dir}" ]] || continue

    log "Copying ${layer} overlay into generated workspace"
    for overlay_entry in "${layer_overlay_dir}"/*; do
        entry_name="$(basename "${overlay_entry}")"
        if [[ "${entry_name}" == "README.md" ]] || [[ "${entry_name}" == "__pycache__" ]]; then
            continue
        fi
        cp -a "${overlay_entry}" "${GENERATED_SRC_DIR}/"
    done
done < <(overlay_layers_for_platform)

if [[ "${PLATFORM}" == "ascend" ]]; then
    log "Materializing prepared kvcache-ops source into generated workspace"
    materialize_kvcache_ops_into_workspace
fi

log "Overlay materialization complete for platform=${PLATFORM}"
