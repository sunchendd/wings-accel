#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOCK_FILE="${PATCH_ROOT}/manifest/lmcache.lock.json"
UPSTREAM_SOURCES_DIR="${PATCH_ROOT}/upstream_sources"
THIRD_PARTY_SOURCES_DIR="${PATCH_ROOT}/third_party_sources"
OVERLAY_DIR="${PATCH_ROOT}/overlay"
PATCHES_ROOT="${PATCH_ROOT}/patches"

BUILD_ROOT="${PATCH_ROOT}/build"
GENERATED_ROOT="${BUILD_ROOT}/generated"
GENERATED_SRC_DIR="${GENERATED_ROOT}/LMCache"
DIST_DIR="${PATCH_ROOT}/dist"
DEFAULT_PLATFORM="common"
PLATFORM="${DEFAULT_PLATFORM}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*"
}

warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $*" >&2
}

fail() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

should_skip_tls_verification_for_url() {
    local url="$1"
    [[ "${url}" == https://artifactrepo.wux-g.tools.xfusion.com/* ]]
}

download_tarball_from_url() {
    local url="$1"
    local dest_path="$2"
    local label="$3"
    local tmp_path="${dest_path}.part"
    local -a curl_args
    local -a wget_args

    [[ -n "${url}" ]] || fail "Missing source_url for ${label}"

    mkdir -p "$(dirname "${dest_path}")"
    rm -f "${tmp_path}"

    curl_args=(-L --fail --retry 3 --retry-delay 2 -o "${tmp_path}")
    wget_args=(-O "${tmp_path}")

    if should_skip_tls_verification_for_url "${url}"; then
        warn "Using insecure TLS download flags for ${label}: ${url}"
        curl_args+=(--insecure)
        wget_args+=(--no-check-certificate)
    fi

    if command -v curl >/dev/null 2>&1; then
        env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
            curl "${curl_args[@]}" "${url}" \
            || fail "Failed to download ${label} tarball from ${url}"
    elif command -v wget >/dev/null 2>&1; then
        env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
            wget "${wget_args[@]}" "${url}" \
            || fail "Failed to download ${label} tarball from ${url}"
    else
        fail "Missing required downloader for ${label}: curl or wget"
    fi

    mv "${tmp_path}" "${dest_path}"
    log "Downloaded ${label} tarball to ${dest_path}"
}

validate_platform() {
    case "${1}" in
        common|ascend)
            ;;
        *)
            fail "Unsupported platform '${1}'. Expected one of: common, ascend"
            ;;
    esac
}

parse_platform_args() {
    PLATFORM="${DEFAULT_PLATFORM}"
    PARSED_ARGS=()
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --platform)
                [[ "$#" -ge 2 ]] || fail "Missing value for --platform"
                PLATFORM="$2"
                shift 2
                ;;
            *)
                PARSED_ARGS+=("$1")
                shift
                ;;
        esac
    done
    validate_platform "${PLATFORM}"
}

read_lock_value() {
    local key="$1"
    python3 - "$LOCK_FILE" "$key" <<'PY'
import json
import sys
from pathlib import Path

lock_path = Path(sys.argv[1])
key = sys.argv[2]
data = json.loads(lock_path.read_text())
value = data
for part in key.split("."):
    if not isinstance(value, dict) or part not in value:
        raise SystemExit(f"Missing key '{key}' in {lock_path}")
    value = value[part]
if isinstance(value, (dict, list)):
    print(json.dumps(value))
else:
    print(value)
PY
}

read_lock_optional_value() {
    local key="$1"
    local default_value="${2:-}"
    python3 - "$LOCK_FILE" "$key" "$default_value" <<'PY'
import json
import sys
from pathlib import Path

lock_path = Path(sys.argv[1])
key = sys.argv[2]
default_value = sys.argv[3]
data = json.loads(lock_path.read_text())
value = data
for part in key.split("."):
    if not isinstance(value, dict) or part not in value:
        print(default_value)
        raise SystemExit(0)
    value = value[part]
if isinstance(value, (dict, list)):
    print(json.dumps(value))
else:
    print(value)
PY
}

tarball_path() {
    local tarball_name
    tarball_name="$(read_lock_value "tarball_name")"
    echo "${UPSTREAM_SOURCES_DIR}/${tarball_name}"
}

is_placeholder_value() {
    local value="$1"
    [[ -z "${value}" ]] || [[ "${value}" == REPLACE_WITH_REAL_SHA256 ]] || [[ "${value}" == REPLACE_WITH_REAL_COMMIT ]]
}

verify_tarball_with_expected_sha() {
    local tarball_path="$1"
    local expected_sha="$2"
    local label="$3"
    local actual_sha

    [[ -f "${tarball_path}" ]] || fail "Missing ${label} tarball: ${tarball_path}"
    if is_placeholder_value "${expected_sha}"; then
        warn "${label} lock does not contain a fixed tarball_sha256 yet; accepting existing tarball at ${tarball_path}"
        return 0
    fi

    actual_sha="$(sha256sum "${tarball_path}" | awk '{print $1}')"
    [[ "${actual_sha}" == "${expected_sha}" ]] || fail "SHA256 mismatch for ${tarball_path}: expected ${expected_sha}, got ${actual_sha}"
}

prepare_git_snapshot_tarball() {
    local label="$1"
    local repo_url="$2"
    local git_ref="$3"
    local root_dir_name="$4"
    local dest_tarball="$5"
    local expected_sha="$6"
    local expected_commit="$7"
    local tmp_root repo_dir resolved_commit actual_sha

    [[ -n "${repo_url}" ]] || fail "Missing internal git URL for ${label}"

    require_cmd git
    require_cmd tar
    require_cmd sha256sum

    tmp_root="$(mktemp -d "${BUILD_ROOT}/.prepare_${label}.XXXXXX")"
    repo_dir="${tmp_root}/repo"

    cleanup_prepare_git_snapshot_tarball() {
        rm -rf "${tmp_root}"
    }

    trap cleanup_prepare_git_snapshot_tarball RETURN

    log "Cloning ${label} from internal git: ${repo_url}"
    git clone "${repo_url}" "${repo_dir}"

    if [[ -n "${git_ref}" ]] && [[ "${git_ref}" != "HEAD" ]] && [[ "${git_ref}" != "default" ]]; then
        if git -C "${repo_dir}" checkout "${git_ref}" >/dev/null 2>&1; then
            log "Checked out ${label} ref ${git_ref}"
        else
            current_branch="$(git -C "${repo_dir}" symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
            if [[ -n "${current_branch}" ]]; then
                warn "Configured ${label} ref '${git_ref}' not found; falling back to cloned default branch '${current_branch}'"
            else
                warn "Configured ${label} ref '${git_ref}' not found; falling back to cloned default HEAD"
            fi
        fi
    else
        current_branch="$(git -C "${repo_dir}" symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
        if [[ -n "${current_branch}" ]]; then
            log "Using ${label} cloned default branch ${current_branch}"
        else
            log "Using ${label} cloned default HEAD"
        fi
    fi

    resolved_commit="$(git -C "${repo_dir}" rev-parse HEAD)"

    if ! is_placeholder_value "${expected_commit}"; then
        [[ "${resolved_commit}" == "${expected_commit}" ]] || fail "Resolved ${label} commit mismatch: expected ${expected_commit}, got ${resolved_commit}"
        log "Prepared ${label} commit matches manifest lock"
    else
        warn "Manifest still contains placeholder ${label} resolved_commit; update the lock with ${resolved_commit}"
    fi

    mkdir -p "$(dirname "${dest_tarball}")"
    rm -f "${dest_tarball}"

    log "Archiving ${label} commit ${resolved_commit} to ${dest_tarball}"
    git -C "${repo_dir}" archive --format=tar.gz --prefix="${root_dir_name}/" -o "${dest_tarball}" HEAD

    actual_sha="$(sha256sum "${dest_tarball}" | awk '{print $1}')"
    log "Prepared ${label} snapshot: commit=${resolved_commit} sha256=${actual_sha}"

    if ! is_placeholder_value "${expected_sha}"; then
        [[ "${actual_sha}" == "${expected_sha}" ]] || fail "Prepared ${label} tarball hash mismatch: expected ${expected_sha}, got ${actual_sha}"
        log "Prepared ${label} tarball matches manifest lock"
    else
        warn "Manifest still contains placeholder ${label} tarball_sha256; update the lock with ${actual_sha}"
    fi

    trap - RETURN
    cleanup_prepare_git_snapshot_tarball
}

verify_locked_tarball() {
    local tarball_path expected_sha
    tarball_path="$(tarball_path)"
    expected_sha="$(read_lock_optional_value "tarball_sha256" "")"

    verify_tarball_with_expected_sha "${tarball_path}" "${expected_sha}" "upstream"
}

download_upstream_source_tarball() {
    local source_url

    source_url="$(read_lock_optional_value "source_url" "")"
    download_tarball_from_url "${source_url}" "$(tarball_path)" "upstream"
}

prepare_upstream_source_from_internal_git() {
    local repo_url git_ref root_dir_name expected_sha expected_commit

    repo_url="$(read_lock_optional_value "internal_git_url" "")"
    git_ref="$(read_lock_optional_value "internal_git_ref" "")"
    root_dir_name="$(read_lock_value "root_dir_name_in_tar")"
    expected_sha="$(read_lock_optional_value "tarball_sha256" "")"
    expected_commit="$(read_lock_optional_value "resolved_commit" "")"

    prepare_git_snapshot_tarball \
        "LMCache" \
        "${repo_url}" \
        "${git_ref}" \
        "${root_dir_name}" \
        "$(tarball_path)" \
        "${expected_sha}" \
        "${expected_commit}"
}

ensure_upstream_source_ready() {
    local source_type tarball

    source_type="$(read_lock_optional_value "source_type" "tarball")"
    tarball="$(tarball_path)"

    case "${source_type}" in
        tarball)
            if [[ -f "${tarball}" ]]; then
                verify_locked_tarball
            else
                download_upstream_source_tarball
                verify_locked_tarball
            fi
            ;;
        internal_git_snapshot)
            if [[ -f "${tarball}" ]]; then
                verify_locked_tarball
            else
                prepare_upstream_source_from_internal_git
                verify_locked_tarball
            fi
            ;;
        *)
            fail "Unsupported source_type '${source_type}'. Expected tarball or internal_git_snapshot"
            ;;
    esac
}

extract_locked_tarball_to() {
    local dest_root="$1"
    local tarball expected_root tmp_dir

    tarball="$(tarball_path)"
    expected_root="$(read_lock_value "root_dir_name_in_tar")"
    verify_locked_tarball

    mkdir -p "${dest_root}"
    tmp_dir="$(mktemp -d "${dest_root}/.extract.XXXXXX")"
    tar -xf "${tarball}" -C "${tmp_dir}"

    mapfile -t top_level_entries < <(
        find "${tmp_dir}" -mindepth 1 -maxdepth 1 \
            ! -name '__MACOSX' \
            ! -name '.DS_Store' \
            -printf '%f\n' | sort
    )

    [[ "${#top_level_entries[@]}" -eq 1 ]] || fail "Expected exactly one top-level directory in tarball, got: ${top_level_entries[*]:-<none>}"
    [[ "${top_level_entries[0]}" == "${expected_root}" ]] || fail "Top-level directory mismatch: expected '${expected_root}', got '${top_level_entries[0]}'"
    [[ -d "${tmp_dir}/${top_level_entries[0]}" ]] || fail "Top-level tarball entry is not a directory: ${top_level_entries[0]}"

    mv "${tmp_dir}/${top_level_entries[0]}" "${dest_root}/LMCache"
    rm -rf "${tmp_dir}"
}

patch_dir_for_locked_version() {
    local layer="$1"
    local version
    version="$(read_lock_value "version")"
    echo "${PATCHES_ROOT}/${layer}/${version}"
}

locked_python_package_version() {
    local version base_version

    version="$(read_lock_value "version")"
    base_version="${version#v}"

    if [[ "${base_version}" == *.dev* ]]; then
        echo "${base_version}"
    else
        echo "${base_version}.dev0"
    fi
}

stamp_generated_lmcache_version_file() {
    local package_version version_tuple version_file

    package_version="$(locked_python_package_version)"
    version_file="${GENERATED_SRC_DIR}/lmcache/_version.py"

    [[ -f "${version_file}" ]] || return 0

    version_tuple="$(python3 - "${package_version}" <<'PY'
import sys

version = sys.argv[1]
parts = []
for part in version.split('.'):
    if part.isdigit():
        parts.append(part)
    else:
        parts.append(repr(part))
print("(" + ", ".join(parts) + ")")
PY
)"

    sed -E -i \
        -e "s/^__version__ = version = .*/__version__ = version = '${package_version}'/" \
        -e "s/^__version_tuple__ = version_tuple = .*/__version_tuple__ = version_tuple = ${version_tuple}/" \
        "${version_file}"

    log "Stamped generated LMCache version file to ${package_version}"
}

patchset_stamp_file() {
    echo "${GENERATED_SRC_DIR}/.wings_patchset_applied.${PLATFORM}"
}

placeholder_patch() {
    local patch_file="$1"
    grep -q '^# PLACEHOLDER PATCH' "${patch_file}"
}

prepare_build_dirs() {
    mkdir -p "${BUILD_ROOT}" "${GENERATED_ROOT}" "${DIST_DIR}" "${THIRD_PARTY_SOURCES_DIR}"
}

ensure_lock_file() {
    [[ -f "${LOCK_FILE}" ]] || fail "Missing lock file: ${LOCK_FILE}"
}

ensure_generated_workspace() {
    [[ -d "${GENERATED_SRC_DIR}" ]] || fail "Generated workspace not found: ${GENERATED_SRC_DIR}. Run unpack-upstream first."
}

clean_generated_workspace() {
    rm -rf "${GENERATED_SRC_DIR}"
}

overlay_layers_for_platform() {
    local layers=("common")
    if [[ "${PLATFORM}" != "common" ]]; then
        layers+=("${PLATFORM}")
    fi
    printf '%s\n' "${layers[@]}"
}

patch_layers_for_platform() {
    overlay_layers_for_platform
}

overlay_dir_for_layer() {
    local layer="$1"
    echo "${OVERLAY_DIR}/${layer}"
}

extra_source_prefix() {
    local source_name="$1"
    echo "extra_sources.${source_name}"
}

extra_source_value() {
    local source_name="$1"
    local field="$2"
    read_lock_value "$(extra_source_prefix "${source_name}").${field}"
}

extra_source_optional_value() {
    local source_name="$1"
    local field="$2"
    local default_value="${3:-}"
    read_lock_optional_value "$(extra_source_prefix "${source_name}").${field}" "${default_value}"
}

extra_source_dir() {
    local source_name="$1"
    local dir_name="${source_name//_/-}"
    echo "${THIRD_PARTY_SOURCES_DIR}/${dir_name}"
}

extra_source_tarball_path() {
    local source_name="$1"
    local tarball_name
    tarball_name="$(extra_source_value "${source_name}" "tarball_name")"
    echo "$(extra_source_dir "${source_name}")/${tarball_name}"
}

extra_source_local_dir_path() {
    local source_name="$1"
    local local_dir_name

    local_dir_name="$(extra_source_optional_value "${source_name}" "local_dir_name" "${source_name//_/-}")"
    echo "$(extra_source_dir "${source_name}")/${local_dir_name}"
}

verify_locked_extra_source_tarball() {
    local source_name="$1"
    local tarball expected_sha actual_sha

    tarball="$(extra_source_tarball_path "${source_name}")"
    expected_sha="$(extra_source_value "${source_name}" "tarball_sha256")"

    verify_tarball_with_expected_sha "${tarball}" "${expected_sha}" "${source_name}"
}

download_extra_source_tarball() {
    local source_name="$1"
    local source_url

    source_url="$(extra_source_optional_value "${source_name}" "source_url" "")"
    download_tarball_from_url \
        "${source_url}" \
        "$(extra_source_tarball_path "${source_name}")" \
        "${source_name}"
}

prepare_extra_source_from_internal_git() {
    local source_name="$1"
    local repo_url git_ref root_dir_name expected_sha expected_commit

    repo_url="$(extra_source_optional_value "${source_name}" "internal_git_url" "")"
    git_ref="$(extra_source_optional_value "${source_name}" "internal_git_ref" "")"
    root_dir_name="$(extra_source_value "${source_name}" "root_dir_name_in_tar")"
    expected_sha="$(extra_source_optional_value "${source_name}" "tarball_sha256" "")"
    expected_commit="$(extra_source_optional_value "${source_name}" "resolved_commit" "")"

    prepare_git_snapshot_tarball \
        "${source_name}" \
        "${repo_url}" \
        "${git_ref}" \
        "${root_dir_name}" \
        "$(extra_source_tarball_path "${source_name}")" \
        "${expected_sha}" \
        "${expected_commit}"
}

ensure_extra_source_ready() {
    local source_name="$1"
    local mode tarball local_dir

    mode="$(extra_source_optional_value "${source_name}" "source_mode" "")"
    [[ -n "${mode}" ]] || fail "Missing lock configuration for extra_sources.${source_name}"
    tarball="$(extra_source_tarball_path "${source_name}")"
    local_dir="$(extra_source_local_dir_path "${source_name}")"

    case "${mode}" in
        tarball)
            if [[ -f "${tarball}" ]]; then
                verify_locked_extra_source_tarball "${source_name}"
            else
                download_extra_source_tarball "${source_name}"
                verify_locked_extra_source_tarball "${source_name}"
            fi
            ;;
        internal_git_snapshot)
            if [[ -f "${tarball}" ]]; then
                verify_locked_extra_source_tarball "${source_name}"
            else
                prepare_extra_source_from_internal_git "${source_name}"
                verify_locked_extra_source_tarball "${source_name}"
            fi
            ;;
        local_directory)
            [[ -d "${local_dir}" ]] || fail "Missing local ${source_name} source directory: ${local_dir}"
            ;;
        *)
            fail "Unsupported ${source_name} source_mode '${mode}'. Expected tarball, internal_git_snapshot, or local_directory"
            ;;
    esac
}

extract_extra_source_tarball_to() {
    local source_name="$1"
    local dest_root="$2"
    local tarball expected_root tmp_dir

    tarball="$(extra_source_tarball_path "${source_name}")"
    expected_root="$(extra_source_value "${source_name}" "root_dir_name_in_tar")"
    verify_locked_extra_source_tarball "${source_name}"

    mkdir -p "${dest_root}"
    tmp_dir="$(mktemp -d "${dest_root}/.${source_name}.extract.XXXXXX")"
    tar -xf "${tarball}" -C "${tmp_dir}"

    mapfile -t top_level_entries < <(
        find "${tmp_dir}" -mindepth 1 -maxdepth 1 \
            ! -name '__MACOSX' \
            ! -name '.DS_Store' \
            -printf '%f\n' | sort
    )

    [[ "${#top_level_entries[@]}" -eq 1 ]] || fail "Expected exactly one top-level directory in ${source_name} tarball, got: ${top_level_entries[*]:-<none>}"
    [[ "${top_level_entries[0]}" == "${expected_root}" ]] || fail "${source_name} top-level directory mismatch: expected '${expected_root}', got '${top_level_entries[0]}'"
    [[ -d "${tmp_dir}/${top_level_entries[0]}" ]] || fail "${source_name} top-level tarball entry is not a directory: ${top_level_entries[0]}"

    mv "${tmp_dir}/${top_level_entries[0]}" "${dest_root}/${expected_root}"
    rm -rf "${tmp_dir}"
}

ensure_kvcache_ops_source_ready() {
    ensure_extra_source_ready "kvcache_ops"
}

materialize_kvcache_ops_into_workspace() {
    local dest_parent="${GENERATED_SRC_DIR}/third_party"
    local root_name mode local_dir extracted_dir final_dir

    ensure_kvcache_ops_source_ready
    mode="$(extra_source_optional_value "kvcache_ops" "source_mode" "")"
    root_name="$(extra_source_value "kvcache_ops" "root_dir_name_in_tar")"
    local_dir="$(extra_source_local_dir_path "kvcache_ops")"
    extracted_dir="${dest_parent}/${root_name}"
    final_dir="${dest_parent}/kvcache-ops"

    mkdir -p "${dest_parent}"
    rm -rf "${final_dir}" "${extracted_dir}"

    case "${mode}" in
        local_directory)
            cp -a "${local_dir}" "${final_dir}"
            ;;
        tarball|internal_git_snapshot)
            extract_extra_source_tarball_to "kvcache_ops" "${dest_parent}"
            if [[ "${root_name}" != "kvcache-ops" ]]; then
                mv "${extracted_dir}" "${final_dir}"
            fi
            ;;
        *)
            fail "Unsupported kvcache_ops source_mode '${mode}' during materialization"
            ;;
    esac
}
