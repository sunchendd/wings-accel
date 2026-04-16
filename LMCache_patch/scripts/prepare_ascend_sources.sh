#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

require_cmd git
require_cmd tar
require_cmd sha256sum
require_cmd python3

parse_platform_args "$@"
[[ "${PLATFORM}" == "ascend" ]] || fail "prepare_ascend_sources.sh only supports --platform ascend"

prepare_build_dirs

SOURCE_MODE="$(extra_source_optional_value "kvcache_ops" "source_mode" "")"
if [[ "${SOURCE_MODE}" == "local_directory" ]]; then
    LOCAL_DIR="$(extra_source_local_dir_path "kvcache_ops")"
    [[ -d "${LOCAL_DIR}" ]] || fail "Missing local kvcache-ops source directory: ${LOCAL_DIR}"
    log "kvcache-ops source_mode=local_directory; using existing local source at ${LOCAL_DIR}"
    exit 0
fi

INTERNAL_GIT_URL="$(extra_source_value "kvcache_ops" "internal_git_url")"
INTERNAL_GIT_REF="$(extra_source_value "kvcache_ops" "internal_git_ref")"
TARBALL_NAME="$(extra_source_value "kvcache_ops" "tarball_name")"
ROOT_DIR_NAME="$(extra_source_value "kvcache_ops" "root_dir_name_in_tar")"
EXPECTED_SHA="$(extra_source_optional_value "kvcache_ops" "tarball_sha256" "")"
EXPECTED_COMMIT="$(extra_source_optional_value "kvcache_ops" "resolved_commit" "")"
DEST_DIR="$(extra_source_dir "kvcache_ops")"
DEST_TARBALL="${DEST_DIR}/${TARBALL_NAME}"

[[ -n "${INTERNAL_GIT_URL}" ]] || fail "Missing internal git URL for kvcache_ops"

TMP_ROOT="$(mktemp -d "${BUILD_ROOT}/.prepare_ascend_sources.XXXXXX")"
REPO_DIR="${TMP_ROOT}/repo"

cleanup() {
    rm -rf "${TMP_ROOT}"
}
trap cleanup EXIT

log "Cloning kvcache-ops from internal git: ${INTERNAL_GIT_URL}"
git clone "${INTERNAL_GIT_URL}" "${REPO_DIR}"

if [[ -n "${INTERNAL_GIT_REF}" ]] && [[ "${INTERNAL_GIT_REF}" != "HEAD" ]] && [[ "${INTERNAL_GIT_REF}" != "default" ]]; then
    if git -C "${REPO_DIR}" checkout "${INTERNAL_GIT_REF}" >/dev/null 2>&1; then
        log "Checked out kvcache-ops ref ${INTERNAL_GIT_REF}"
    else
        CURRENT_BRANCH="$(git -C "${REPO_DIR}" symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
        if [[ -n "${CURRENT_BRANCH}" ]]; then
            warn "Configured kvcache-ops ref '${INTERNAL_GIT_REF}' not found; falling back to cloned default branch '${CURRENT_BRANCH}'"
        else
            warn "Configured kvcache-ops ref '${INTERNAL_GIT_REF}' not found; falling back to cloned default HEAD"
        fi
    fi
else
    CURRENT_BRANCH="$(git -C "${REPO_DIR}" symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
    if [[ -n "${CURRENT_BRANCH}" ]]; then
        log "Using kvcache-ops cloned default branch ${CURRENT_BRANCH}"
    else
        log "Using kvcache-ops cloned default HEAD"
    fi
fi

RESOLVED_COMMIT="$(git -C "${REPO_DIR}" rev-parse HEAD)"

if [[ -n "${EXPECTED_COMMIT}" ]] && [[ "${EXPECTED_COMMIT}" != "REPLACE_WITH_REAL_COMMIT" ]]; then
    if [[ "${RESOLVED_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
        fail "Resolved kvcache-ops commit mismatch: expected ${EXPECTED_COMMIT}, got ${RESOLVED_COMMIT}"
    fi
    log "Prepared kvcache-ops commit matches manifest lock"
else
    warn "Manifest still contains placeholder kvcache_ops resolved_commit; update the lock with ${RESOLVED_COMMIT}"
fi

mkdir -p "${DEST_DIR}"
rm -f "${DEST_TARBALL}"

log "Archiving kvcache-ops commit ${RESOLVED_COMMIT} to ${DEST_TARBALL}"
git -C "${REPO_DIR}" archive --format=tar.gz --prefix="${ROOT_DIR_NAME}/" -o "${DEST_TARBALL}" HEAD

ACTUAL_SHA="$(sha256sum "${DEST_TARBALL}" | awk '{print $1}')"
log "Prepared kvcache-ops snapshot: commit=${RESOLVED_COMMIT} sha256=${ACTUAL_SHA}"

if [[ -n "${EXPECTED_SHA}" ]] && [[ "${EXPECTED_SHA}" != "REPLACE_WITH_REAL_SHA256" ]]; then
    if [[ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]]; then
        fail "Prepared kvcache-ops tarball hash mismatch: expected ${EXPECTED_SHA}, got ${ACTUAL_SHA}"
    fi
    log "Prepared kvcache-ops tarball matches manifest lock"
else
    warn "Manifest still contains placeholder kvcache_ops tarball_sha256; update the lock with ${ACTUAL_SHA}"
fi
