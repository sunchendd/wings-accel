#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

require_cmd git
require_cmd tar
require_cmd sha256sum
require_cmd python3

parse_platform_args "$@"
[[ "${PLATFORM}" == "common" ]] || fail "prepare_common_sources.sh only supports --platform common"

prepare_build_dirs

ensure_upstream_source_ready
ensure_extra_source_ready "qatzip"

log "Prepared common-path LMCache sources successfully"