#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_ROOT="${ROOT_DIR}/LMCache_patch"
OUTPUT_ROOT_DEFAULT="${ROOT_DIR}/build/output/lmcache"
OUTPUT_ROOT="${OUTPUT_ROOT_DEFAULT}"
MANIFEST_PATH="${ROOT_DIR}/build/output/lmcache_manifest.json"

NVIDIA_X86_IMAGE_DEFAULT="docker.artifactrepo.wux-g.tools.xfusion.com/ai_solution/ci/wings/x86/vllm-openai_cmake_3.30.3_full_cuda12.9:v0.17.0"
ASCEND_ARM_IMAGE_DEFAULT="docker.artifactrepo.wux-g.tools.xfusion.com/ai_solution/ci/wings/arm/quay.io/ascend/vllm-ascend_sshkey:v0.17.0rc1"

default_targets_for_arch() {
    case "$(uname -m)" in
        x86_64)
            echo "nvidia-x86"
            ;;
        aarch64)
            echo "ascend-arm"
            ;;
        *)
            fail "Unsupported host architecture '$(uname -m)' for LMCache build"
            ;;
    esac
}

TARGETS_CSV="${WINGS_LMCACHE_TARGETS:-$(default_targets_for_arch)}"

log() {
    echo "[wings-accel] $*"
}

fail() {
    echo "[wings-accel] Error: $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: build/build_lmcache.sh [--outdir DIR] [--manifest-path FILE] [--targets TARGETS]

TARGETS is a comma-separated subset of:
    nvidia-x86,ascend-arm

Environment overrides:
  WINGS_LMCACHE_NVIDIA_X86_IMAGE
  WINGS_LMCACHE_ASCEND_ARM_IMAGE
  WINGS_LMCACHE_TARGETS
    WINGS_LMCACHE_QAT_PACKAGE_ROOT
    WINGS_LMCACHE_QAT_RUNTIME_ROOT
    WINGS_LMCACHE_CUDA_HOME
EOF
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --outdir)
            [[ "$#" -ge 2 ]] || fail "Missing value for --outdir"
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --targets)
            [[ "$#" -ge 2 ]] || fail "Missing value for --targets"
            TARGETS_CSV="$2"
            shift 2
            ;;
        --manifest-path)
            [[ "$#" -ge 2 ]] || fail "Missing value for --manifest-path"
            MANIFEST_PATH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "Unknown argument: $1"
            ;;
    esac
done

command -v docker >/dev/null 2>&1 || fail "docker is required for LMCache matrix builds"
[[ -d "${PATCH_ROOT}" ]] || fail "LMCache_patch not found at ${PATCH_ROOT}"

mkdir -p "${OUTPUT_ROOT}"

IFS=',' read -r -a TARGETS <<< "${TARGETS_CSV}"

validate_target() {
    case "$1" in
        nvidia-x86|ascend-arm)
            ;;
        *)
            fail "Unsupported LMCache target '$1'"
            ;;
    esac
}

builder_image_for_target() {
    case "$1" in
        nvidia-x86)
            echo "${WINGS_LMCACHE_NVIDIA_X86_IMAGE:-${NVIDIA_X86_IMAGE_DEFAULT}}"
            ;;
        ascend-arm)
            echo "${WINGS_LMCACHE_ASCEND_ARM_IMAGE:-${ASCEND_ARM_IMAGE_DEFAULT}}"
            ;;
    esac
}

container_name_for_target() {
    case "$1" in
        nvidia-x86)
            echo "wings-lmcache-nvidia-x86-builder-$$"
            ;;
        ascend-arm)
            echo "wings-lmcache-ascend-arm-builder-$$"
            ;;
    esac
}

build_command_for_target() {
    case "$1" in
        nvidia-x86)
            cat <<'EOF'
set -euo pipefail

find_cuda_header_dir() {
    local candidate
    for candidate in \
        "${CUDA_HOME:-}/include" \
        "${CUDA_HOME:-}/targets/x86_64-linux/include" \
        /usr/local/cuda/include \
        /usr/local/cuda/targets/x86_64-linux/include \
        /opt/wings-cuda/include \
        /opt/wings-cuda/targets/x86_64-linux/include; do
        [[ -n "${candidate}" ]] || continue
        if [[ -f "${candidate}/cusparse.h" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

prepend_env_path() {
    local var_name="$1"
    local entry="$2"
    local current_value="${!var_name:-}"
    [[ -n "${entry}" ]] || return 0
    if [[ -z "${current_value}" ]]; then
        export "${var_name}=${entry}"
    elif [[ ":${current_value}:" != *":${entry}:"* ]]; then
        export "${var_name}=${entry}:${current_value}"
    fi
}

if [[ -d /opt/wings-cuda ]]; then
    export CUDA_HOME=/opt/wings-cuda
    prepend_env_path CPATH /opt/wings-cuda/include
    prepend_env_path CPATH /opt/wings-cuda/targets/x86_64-linux/include
    prepend_env_path C_INCLUDE_PATH /opt/wings-cuda/include
    prepend_env_path C_INCLUDE_PATH /opt/wings-cuda/targets/x86_64-linux/include
    prepend_env_path CPLUS_INCLUDE_PATH /opt/wings-cuda/include
    prepend_env_path CPLUS_INCLUDE_PATH /opt/wings-cuda/targets/x86_64-linux/include
    prepend_env_path LIBRARY_PATH /opt/wings-cuda/lib64
    prepend_env_path LIBRARY_PATH /opt/wings-cuda/targets/x86_64-linux/lib
    prepend_env_path LIBRARY_PATH /opt/wings-cuda/targets/x86_64-linux/lib/stubs
    prepend_env_path LD_LIBRARY_PATH /opt/wings-cuda/lib64
    prepend_env_path LD_LIBRARY_PATH /opt/wings-cuda/targets/x86_64-linux/lib
fi

if ! cuda_include_dir="$(find_cuda_header_dir)"; then
    echo "[wings-accel] Error: Missing CUDA development header cusparse.h for NVIDIA LMCache build. This is not a QATzip issue. Use a CUDA devel image or set WINGS_LMCACHE_CUDA_HOME to a host CUDA toolkit root that contains include/cusparse.h." >&2
    exit 1
fi

prepend_env_path CPATH "${cuda_include_dir}"
prepend_env_path C_INCLUDE_PATH "${cuda_include_dir}"
prepend_env_path CPLUS_INCLUDE_PATH "${cuda_include_dir}"

cd /tmp/wings-lmcache/LMCache_patch
if [[ -d /tmp/wings-lmcache/QAT ]]; then
    export QAT_PACKAGE_ROOT=/tmp/wings-lmcache/QAT
fi
if [[ -d /tmp/wings-lmcache/wings-qat/include ]]; then
    export QATZIP_INCLUDE_DIR=/tmp/wings-lmcache/wings-qat/include
fi
if [[ -d /tmp/wings-lmcache/wings-qat/lib ]]; then
    export QATZIP_LIB_DIR=/tmp/wings-lmcache/wings-qat/lib
    export LD_LIBRARY_PATH=/tmp/wings-lmcache/wings-qat/lib:${LD_LIBRARY_PATH:-}
fi
python3 install.py build-wheel
EOF
            ;;
        ascend-arm)
            cat <<'EOF'
set -euo pipefail
cd /tmp/wings-lmcache/LMCache_patch
python3 install.py prepare-ascend-sources
python3 install.py build-wheel --platform ascend
EOF
            ;;
    esac
}

copy_host_dir_to_path() {
    local container_name="$1"
    local host_dir="$2"
    local container_dir="$3"

    if [[ -d "${host_dir}" ]]; then
        docker exec "${container_name}" mkdir -p "${container_dir}"
        docker cp "${host_dir}/." "${container_name}:${container_dir}"
    fi
}

require_existing_dir() {
    local dir_path="$1"
    local description="$2"

    [[ -d "${dir_path}" ]] || fail "${description} not found: ${dir_path}"
}

docker_ssh_args() {
    local home_dir known_hosts config_file

    home_dir="${HOME:-}"
    known_hosts="${home_dir}/.ssh/known_hosts"
    config_file="${home_dir}/.ssh/config"

    if [[ -n "${SSH_AUTH_SOCK:-}" ]] && [[ -S "${SSH_AUTH_SOCK}" ]]; then
        printf '%s\n' "-e" "SSH_AUTH_SOCK=/tmp/ssh-agent.sock" "-v" "${SSH_AUTH_SOCK}:/tmp/ssh-agent.sock"
    fi

    if [[ -f "${known_hosts}" ]]; then
        printf '%s\n' "-v" "${known_hosts}:/root/.ssh/known_hosts:ro"
    fi

    if [[ -f "${config_file}" ]]; then
        printf '%s\n' "-v" "${config_file}:/root/.ssh/config:ro"
    fi
}

docker_cuda_args() {
    local cuda_home

    cuda_home="${WINGS_LMCACHE_CUDA_HOME:-}"
    if [[ -n "${cuda_home}" ]]; then
        require_existing_dir "${cuda_home}" "WINGS_LMCACHE_CUDA_HOME"
        printf '%s\n' "-v" "${cuda_home}:/opt/wings-cuda:ro"
    fi
}

download_target_dependency_wheels() {
    local target="$1"
    local target_dir="$2"
    local python_version="$3"
    local primary_wheel
    local deps_dir

    primary_wheel="$(find "${target_dir}" -maxdepth 1 -type f -name 'lmcache-*.whl' | sort | head -n 1)"
    [[ -n "${primary_wheel}" ]] || fail "No lmcache wheel available for dependency download in ${target_dir}"

    deps_dir="${target_dir}/deps"

    rm -rf "${deps_dir}"
    mkdir -p "${deps_dir}"

    log "── Downloading LMCache dependency wheels for '${target}' ..."
    if [[ "${target}" == "ascend-arm" ]]; then
        if ! pip3 download \
            --no-deps \
            aiofile==3.9.0 awscrt==0.32.0 caio==0.9.25 cuda-pathfinder==1.5.2 cufile-python==0.2.0 cupy-cuda12x==14.0.1  nixl==1.0.0 nixl-cu12==1.0.0 numpy==2.2.6 nvtx==0.2.15 redis==7.4.0 sortedcontainers==2.4.0 \
            --dest "${deps_dir}" \
            --python-version "${python_version}" \
            --only-binary=:all: ; then
            fail "Failed to download LMCache dependency wheels for target '${target}'"
        fi
     elif [[ "${target}" == "nvidia-x86" ]]; then
        if ! pip3 download \
            --no-deps \
            aiofile==3.9.0 awscrt==0.32.0 caio==0.9.25 cuda-pathfinder==1.5.2 cufile-python==0.2.0 cupy-cuda12x==14.0.1  nixl==1.0.0  numpy==2.2.6 nvtx==0.2.15 redis==7.4.0 sortedcontainers==2.4.0 \
            --dest "${deps_dir}" \
            --python-version "${python_version}" \
            --only-binary=:all: ; then
            fail "Failed to download LMCache dependency wheels for target '${target}'"
        fi
     else
         exit 1
     fi
     
    if ! find "${deps_dir}" -maxdepth 1 -type f -name '*.whl' | grep -q .; then
        fail "No LMCache dependency wheels were downloaded for target '${target}'"
    fi
}

copy_target_artifacts() {
    local container_name="$1"
    local target_dir="$2"

    rm -rf "${target_dir}"
    mkdir -p "${target_dir}"
    docker cp "${container_name}:/tmp/wings-lmcache/LMCache_patch/dist/." "${target_dir}/"

    if ! find "${target_dir}" -maxdepth 1 -type f -name 'lmcache-*.whl' | grep -q .; then
        fail "No lmcache wheel produced in ${target_dir}"
    fi
}

build_target() {
    local target="$1"
    local image
    local container_name
    local build_script
    local target_dir
    local -a docker_run_args

    validate_target "${target}"
    image="$(builder_image_for_target "${target}")"
    container_name="$(container_name_for_target "${target}")"
    build_script="$(build_command_for_target "${target}")"
    target_dir="${OUTPUT_ROOT}/${target}"

    log "── Building LMCache target '${target}' with image ${image} ..."
    docker rm -f "${container_name}" >/dev/null 2>&1 || true
    docker_run_args=(--name "${container_name}" -d)
    while IFS= read -r arg; do
        [[ -n "${arg}" ]] || continue
        docker_run_args+=("${arg}")
    done < <(docker_ssh_args)
    while IFS= read -r arg; do
        [[ -n "${arg}" ]] || continue
        docker_run_args+=("${arg}")
    done < <(docker_cuda_args)
    docker run "${docker_run_args[@]}" "${image}" sleep 3600 >/dev/null

    cleanup_target() {
        docker rm -f "${container_name}" >/dev/null 2>&1 || true
    }

    trap cleanup_target RETURN

    docker exec "${container_name}" mkdir -p /tmp/wings-lmcache
    docker cp "${PATCH_ROOT}" "${container_name}:/tmp/wings-lmcache"
    docker exec "${container_name}" /bin/bash -lc "chmod -R a+rX /tmp/wings-lmcache/LMCache_patch && find /tmp/wings-lmcache/LMCache_patch/scripts -type f -name '*.sh' -exec chmod 755 {} +"
    if [[ "${target}" == "nvidia-x86" ]]; then
        if [[ -n "${WINGS_LMCACHE_QAT_PACKAGE_ROOT:-}" ]]; then
            require_existing_dir "${WINGS_LMCACHE_QAT_PACKAGE_ROOT}" "WINGS_LMCACHE_QAT_PACKAGE_ROOT"
            copy_host_dir_to_path "${container_name}" "${WINGS_LMCACHE_QAT_PACKAGE_ROOT}" "/tmp/wings-lmcache/QAT"
        fi
        if [[ -n "${WINGS_LMCACHE_QAT_RUNTIME_ROOT:-}" ]]; then
            require_existing_dir "${WINGS_LMCACHE_QAT_RUNTIME_ROOT}" "WINGS_LMCACHE_QAT_RUNTIME_ROOT"
            copy_host_dir_to_path "${container_name}" "${WINGS_LMCACHE_QAT_RUNTIME_ROOT}" "/tmp/wings-lmcache/wings-qat"
        fi
    fi
    docker exec "${container_name}" /bin/bash -lc "${build_script}"
    copy_target_artifacts "${container_name}" "${target_dir}"
    if [[ "${target}" == "nvidia-x86" ]]; then
        download_target_dependency_wheels "${target}" "${target_dir}" "3.12"
        wget -P "${target_dir}/deps"  https://mirrors.xfusion.com/pypi/packages/48/68/f58b0b1aa8d2d03dd8354f6893fa858c77267ddef6cdd2d20868cf0ea88b/nixl_cu12-1.0.0-cp312-cp312-manylinux_2_28_x86_64.whl
    elif [[ "${target}" == "ascend-arm" ]]; then
        download_target_dependency_wheels "${target}" "${target_dir}" "3.11"
    else
        exit 1
    fi

    trap - RETURN
    cleanup_target
    log "✅ LMCache target '${target}' outputs written to ${target_dir}"
}

for target in "${TARGETS[@]}"; do
    build_target "${target}"
done

python3 "${SCRIPT_DIR}/generate_lmcache_manifest.py" \
    --output-root "${OUTPUT_ROOT}" \
    --manifest-path "${MANIFEST_PATH}"

log "✅ LMCache manifest written to ${MANIFEST_PATH}"
