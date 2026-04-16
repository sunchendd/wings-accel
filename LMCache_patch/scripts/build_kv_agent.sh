#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

parse_platform_args "$@"
prepare_build_dirs

SKIP_PREPARE="false"
OPTIONAL_MODE="false"
for arg in "${PARSED_ARGS[@]}"; do
    case "${arg}" in
        --skip-prepare)
            SKIP_PREPARE="true"
            ;;
        --optional)
            OPTIONAL_MODE="true"
            ;;
    esac
done

optional_skip_or_fail() {
    local message="$1"
    if [[ "${OPTIONAL_MODE}" == "true" ]] && [[ "${WINGS_STRICT_QAT_BUILD:-0}" != "1" ]]; then
        warn "${message}"
        exit 0
    fi
    fail "${message}"
}

require_cmd_or_skip() {
    local cmd_name="$1"
    if ! command -v "${cmd_name}" >/dev/null 2>&1; then
        optional_skip_or_fail "Missing required command for kv-agent build: ${cmd_name}"
    fi
}

require_cmd python3
require_cmd_or_skip gcc
require_cmd_or_skip g++
require_cmd_or_skip make

if [[ "${PLATFORM}" == "ascend" ]]; then
    fail "kv-agent build is not supported for platform=ascend because Ascend workflow explicitly disables QAT"
fi

HOST_ARCH="$(uname -m)"
case "${HOST_ARCH}" in
    aarch64|arm64)
        optional_skip_or_fail \
            "Skipping kv-agent build on host arch ${HOST_ARCH}: Wings QAT is optional on ARM and should degrade to the raw local-disk path"
        ;;
esac

if [[ "${SKIP_PREPARE}" != "true" ]]; then
    "${SCRIPT_DIR}/unpack_upstream.sh" --platform "${PLATFORM}"
    "${SCRIPT_DIR}/materialize_workspace.sh" --platform "${PLATFORM}"
    "${SCRIPT_DIR}/apply_patchset.sh" --platform "${PLATFORM}" --force
fi

ensure_generated_workspace

KV_AGENT_ROOT="${GENERATED_SRC_DIR}/third_party/kv-agent"
[[ -d "${KV_AGENT_ROOT}" ]] || fail "Vendored kv-agent source not found: ${KV_AGENT_ROOT}"

QZIP_DIR="${KV_AGENT_ROOT}/qzip"
LIB_DIR="${KV_AGENT_ROOT}/lib"
QAT_STAGE_ROOT="${BUILD_ROOT}/qatzip-stage"
QAT_STAGE_INCLUDE_DIR="${QAT_STAGE_ROOT}/include"
QAT_STAGE_LIB_DIR="${QAT_STAGE_ROOT}/lib"
QATZIP_INSTALL_PREFIX="${QATZIP_INSTALL_PREFIX:-${BUILD_ROOT}/qatzip-install}"
QATZIP_SOURCE_BUILD_ROOT="${BUILD_ROOT}/third_party_sources/qatzip"

prepend_env_path() {
    local var_name="$1"
    local entry="$2"
    local current_value="${!var_name:-}"
    if [[ -z "${entry}" ]]; then
        return 0
    fi
    if [[ -z "${current_value}" ]]; then
        export "${var_name}=${entry}"
    elif [[ ":${current_value}:" != *":${entry}:"* ]]; then
        export "${var_name}=${entry}:${current_value}"
    fi
}

find_header_dir() {
    local header_name="$1"
    local candidate
    for candidate in \
        "${QATZIP_INCLUDE_DIR:-}" \
        /opt/wings-qat/include \
        /usr/include \
        /usr/local/include \
        /usr/include/qatzip \
        /usr/local/include/qatzip \
        /opt/intel/QATzip/include; do
        [[ -n "${candidate}" ]] || continue
        if [[ -f "${candidate}/${header_name}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

find_explicit_qat_package_root() {
    local candidate="${QAT_PACKAGE_ROOT:-}"
    if [[ -n "${candidate}" ]] && [[ -d "${candidate}" ]]; then
        echo "${candidate}"
        return 0
    fi
    return 1
}

find_shared_lib() {
    local lib_pattern="$1"
    local search_dir candidate
    for search_dir in \
        "${QATZIP_LIB_DIR:-}" \
        /opt/wings-qat/lib \
        "${QAT_PACKAGE_ROOT:-}" \
        "${QAT_PACKAGE_ROOT:-}/build" \
        /usr/lib \
        /usr/lib/x86_64-linux-gnu \
        /usr/lib64 \
        /usr/local/lib \
        /lib \
        /lib/x86_64-linux-gnu \
        /lib64; do
        [[ -n "${search_dir}" ]] || continue
        for candidate in "${search_dir}"/${lib_pattern}; do
            if [[ -e "${candidate}" ]]; then
                echo "${candidate}"
                return 0
            fi
        done
    done
    return 1
}

copy_runtime_lib() {
    local link_name="$1"
    local lib_pattern="$2"
    local resolved
    resolved="$(find_shared_lib "${lib_pattern}")" || fail "Missing required QAT runtime library: ${link_name}"
    cp -a "${resolved}" "${LIB_DIR}/"
    if [[ ! -e "${LIB_DIR}/${link_name}" ]]; then
        ln -sf "$(basename "${resolved}")" "${LIB_DIR}/${link_name}"
    fi
}

ensure_qatzip_source_ready() {
    ensure_extra_source_ready "qatzip"
}

materialize_qatzip_source() {
    local root_name

    ensure_qatzip_source_ready
    root_name="$(extra_source_value "qatzip" "root_dir_name_in_tar")"

    rm -rf "${QATZIP_SOURCE_BUILD_ROOT}"
    mkdir -p "${QATZIP_SOURCE_BUILD_ROOT}"
    extract_extra_source_tarball_to "qatzip" "${QATZIP_SOURCE_BUILD_ROOT}"
    echo "${QATZIP_SOURCE_BUILD_ROOT}/${root_name}"
}

build_qatzip_from_locked_snapshot() {
    local qatzip_source_dir qat_package_root

    qatzip_source_dir="$(materialize_qatzip_source)"
    qat_package_root="$(find_explicit_qat_package_root)" || optional_skip_or_fail \
        "Missing QAT_PACKAGE_ROOT. Provide an explicit offline QAT package tree when building QATzip from third_party_sources/qatzip."

    require_cmd_or_skip autoconf
    require_cmd_or_skip automake
    require_cmd_or_skip libtool
    require_cmd_or_skip pkg-config

    log "QATzip runtime not fully detected; building from locked offline snapshot ${qatzip_source_dir}"
    rm -rf "${QATZIP_INSTALL_PREFIX}"
    mkdir -p "${QATZIP_INSTALL_PREFIX}"

    pushd "${qatzip_source_dir}" >/dev/null

    if [[ -f "autogen.sh" ]]; then
        QZ_ROOT="${qatzip_source_dir}" ICP_ROOT="${qat_package_root}" ./autogen.sh
    fi

    if [[ -f "configure" ]]; then
        QZ_ROOT="${qatzip_source_dir}" ICP_ROOT="${qat_package_root}" ./configure --prefix="${QATZIP_INSTALL_PREFIX}"
    else
        popd >/dev/null
        return 1
    fi

    make -j"$(nproc)"
    make install
    popd >/dev/null

    export QATZIP_INCLUDE_DIR="${QATZIP_INSTALL_PREFIX}/include"
    export QATZIP_LIB_DIR="${QATZIP_INSTALL_PREFIX}/lib"
    return 0
}

stage_qat_runtime_artifacts() {
    local qatzip_header_dir qatzip_so_path qat_s_so_path usdm_so_path numa_so_path

    qatzip_header_dir="$(find_header_dir "qatzip.h")" || optional_skip_or_fail \
        "Missing required header qatzip.h. Install QATzip headers, set QATZIP_INCLUDE_DIR, or place the locked QATzip tarball under third_party_sources/qatzip and set QAT_PACKAGE_ROOT."
    qatzip_so_path="$(find_shared_lib "libqatzip.so*")" || optional_skip_or_fail \
        "Missing required shared library libqatzip.so. Install QATzip runtime, set QATZIP_LIB_DIR, or place the locked QATzip tarball under third_party_sources/qatzip and set QAT_PACKAGE_ROOT."
    qat_s_so_path="$(find_shared_lib "libqat_s.so*")" || optional_skip_or_fail \
        "Missing required shared library libqat_s.so. Install QAT runtime or set QAT_PACKAGE_ROOT/QATZIP_LIB_DIR explicitly."
    usdm_so_path="$(find_shared_lib "libusdm_drv_s.so*")" || optional_skip_or_fail \
        "Missing required shared library libusdm_drv_s.so. Install QAT runtime or set QAT_PACKAGE_ROOT/QATZIP_LIB_DIR explicitly."
    numa_so_path="$(find_shared_lib "libnuma.so*")" || optional_skip_or_fail \
        "Missing required shared library libnuma.so. Install libnuma development/runtime packages."

    rm -rf "${QAT_STAGE_ROOT}"
    mkdir -p "${QAT_STAGE_INCLUDE_DIR}" "${QAT_STAGE_LIB_DIR}"

    cp -a "${qatzip_header_dir}/qatzip.h" "${QAT_STAGE_INCLUDE_DIR}/"
    cp -a "${qatzip_so_path}" "${QAT_STAGE_LIB_DIR}/"
    cp -a "${qat_s_so_path}" "${QAT_STAGE_LIB_DIR}/"
    cp -a "${usdm_so_path}" "${QAT_STAGE_LIB_DIR}/"
    cp -a "${numa_so_path}" "${QAT_STAGE_LIB_DIR}/"

    export QATZIP_INCLUDE_DIR="${QAT_STAGE_INCLUDE_DIR}"
    export QATZIP_LIB_DIR="${QAT_STAGE_LIB_DIR}"
}

log "Preparing kv-agent runtime libraries"
mkdir -p "${LIB_DIR}"

[[ -f "${KV_AGENT_ROOT}/setup.py" ]] || optional_skip_or_fail "Missing vendored kv-agent setup.py: ${KV_AGENT_ROOT}/setup.py"
[[ -f "${QZIP_DIR}/Makefile" ]] || optional_skip_or_fail "Missing qzip build file: ${QZIP_DIR}/Makefile"

if ! find_header_dir "qatzip.h" >/dev/null 2>&1 || ! find_shared_lib "libqatzip.so*" >/dev/null 2>&1; then
    build_qatzip_from_locked_snapshot || true
fi

stage_qat_runtime_artifacts

QATZIP_HEADER_DIR="$(find_header_dir "qatzip.h")"
QATZIP_SO_PATH="$(find_shared_lib "libqatzip.so*")"
QAT_S_SO_PATH="$(find_shared_lib "libqat_s.so*")"
USDM_SO_PATH="$(find_shared_lib "libusdm_drv_s.so*")"
NUMA_SO_PATH="$(find_shared_lib "libnuma.so*")"

prepend_env_path "CPATH" "${QATZIP_HEADER_DIR}"
prepend_env_path "C_INCLUDE_PATH" "${QATZIP_HEADER_DIR}"
prepend_env_path "CPLUS_INCLUDE_PATH" "${QATZIP_HEADER_DIR}"
prepend_env_path "LIBRARY_PATH" "$(dirname "${QATZIP_SO_PATH}")"
prepend_env_path "LIBRARY_PATH" "$(dirname "${QAT_S_SO_PATH}")"
prepend_env_path "LIBRARY_PATH" "$(dirname "${USDM_SO_PATH}")"
prepend_env_path "LIBRARY_PATH" "$(dirname "${NUMA_SO_PATH}")"
prepend_env_path "LD_LIBRARY_PATH" "$(dirname "${QATZIP_SO_PATH}")"
prepend_env_path "LD_LIBRARY_PATH" "$(dirname "${QAT_S_SO_PATH}")"
prepend_env_path "LD_LIBRARY_PATH" "$(dirname "${USDM_SO_PATH}")"
prepend_env_path "LD_LIBRARY_PATH" "$(dirname "${NUMA_SO_PATH}")"

log "Using QATzip headers from ${QATZIP_HEADER_DIR}"
log "Using QAT runtime libraries from $(dirname "${QATZIP_SO_PATH}")"
log "Staged QAT runtime artifacts under ${QAT_STAGE_ROOT}"

log "Building auxiliary libqzip.so"
make -C "${QZIP_DIR}" clean
if ! make -C "${QZIP_DIR}"; then
    optional_skip_or_fail "Failed to build auxiliary libqzip.so from ${QZIP_DIR}"
fi
cp -a "${QZIP_DIR}/libqzip.so" "${LIB_DIR}/"
copy_runtime_lib "libqatzip.so" "libqatzip.so*"
copy_runtime_lib "libqat_s.so" "libqat_s.so*"
copy_runtime_lib "libusdm_drv_s.so" "libusdm_drv_s.so*"
copy_runtime_lib "libnuma.so.1" "libnuma.so*"

pushd "${KV_AGENT_ROOT}" >/dev/null

log "Building kv-agent extension in place"
if ! python3 setup.py build_ext --inplace; then
    popd >/dev/null
    optional_skip_or_fail "Failed to build kv-agent extension in place"
fi

log "Building kv-agent wheel"
if ! python3 setup.py bdist_wheel --dist-dir "${DIST_DIR}"; then
    popd >/dev/null
    optional_skip_or_fail "Failed to build kv-agent wheel"
fi

popd >/dev/null

log "kv-agent build complete"
find "${DIST_DIR}" -maxdepth 1 -type f -name 'kv_agent-*.whl' -print
