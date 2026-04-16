#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

require_cmd python3
require_cmd tar
require_cmd sha256sum
parse_platform_args "$@"
ensure_lock_file
prepare_build_dirs
ensure_upstream_source_ready

log "Cleaning generated workspace"
clean_generated_workspace

log "Unpacking $(tarball_path)"
extract_locked_tarball_to "${GENERATED_ROOT}"
stamp_generated_lmcache_version_file
log "Generated workspace ready at ${GENERATED_SRC_DIR} for platform=${PLATFORM}"
