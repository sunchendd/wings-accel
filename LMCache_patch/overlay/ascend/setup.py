# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import configparser
import glob
import os
import platform
import shutil
import subprocess
import sys

# Third Party
from setuptools import find_packages, setup

ROOT_DIR = Path(__file__).parent
HIPIFY_DIR = os.path.join(ROOT_DIR, "csrc/")
HIPIFY_OUT_DIR = os.path.join(ROOT_DIR, "csrc_hip/")

# python -m build --sdist
# will run python setup.py sdist --dist-dir dist
BUILDING_SDIST = "sdist" in sys.argv or os.environ.get("NO_CUDA_EXT", "0") == "1"

# New environment variable to choose between CUDA and HIP
BUILD_WITH_HIP = os.environ.get("BUILD_WITH_HIP", "0") == "1"
BUILD_WITH_ASCEND = os.environ.get("BUILD_WITH_ASCEND", "0") == "1"

ENABLE_CXX11_ABI = os.environ.get("ENABLE_CXX11_ABI", "1") == "1"


def _ascend_arch_subdir() -> str:
    machine = platform.machine()
    if machine == "aarch64":
        return "aarch64-linux"
    if machine == "x86_64":
        return "x86_64-linux"
    raise RuntimeError(f"Unsupported Ascend build architecture: {machine}")


def _get_ascend_home_path() -> Path:
    return Path(
        os.environ.get(
            "ASCEND_HOME_PATH",
            "/usr/local/Ascend/ascend-toolkit/latest",
        )
    )


def _get_soc_version() -> str:
    preset = os.environ.get("SOC_VERSION")
    if preset:
        return preset

    try:
        output = subprocess.check_output(
            ["npu-smi", "info", "-t", "board", "-i", "0", "-c", "0"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Unable to determine Ascend SOC version. "
            "Set SOC_VERSION explicitly before BUILD_WITH_ASCEND=1 builds."
        ) from exc

    board_info = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        board_info[key.strip()] = value.strip()

    chip_name = board_info.get("Chip Name")
    npu_name = board_info.get("NPU Name")
    if not chip_name:
        raise RuntimeError(
            "Unable to parse 'Chip Name' from npu-smi output. "
            "Set SOC_VERSION explicitly before BUILD_WITH_ASCEND=1 builds."
        )

    if npu_name:
        return f"{chip_name}_{npu_name}"
    return chip_name if chip_name.startswith("Ascend") else f"Ascend{chip_name}"


def _get_aicore_arch_number(ascend_home: Path, soc_version: str) -> str | None:
    override = os.environ.get("ASCEND_AICORE_ARCH")
    if override:
        return override

    platform_config = (
        ascend_home / _ascend_arch_subdir() / "data" / "platform_config" / f"{soc_version}.ini"
    )
    if not platform_config.exists():
        return None

    parser = configparser.ConfigParser()
    parser.read(platform_config)
    aic_version = parser.get("version", "AIC_version", fallback="")
    if not aic_version:
        return None
    return aic_version.split("-")[-1]


def _get_torch_npu_path() -> Path:
    try:
        import torch_npu
    except ImportError as exc:
        raise RuntimeError(
            "BUILD_WITH_ASCEND=1 requires torch_npu to be installed in the build environment."
        ) from exc

    return Path(torch_npu.__file__).resolve().parent


def _write_kvcache_ops_wrapper(wrapper_dir: Path) -> Path:
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path = wrapper_dir / "CMakeLists.txt"
    wrapper_path.write_text(
        """
cmake_minimum_required(VERSION 3.16)
project(kvcache_ops_wrapper LANGUAGES CXX)

if(NOT DEFINED KVCACHE_OPS_SOURCE_DIR)
  message(FATAL_ERROR "KVCACHE_OPS_SOURCE_DIR is required")
endif()

if(NOT DEFINED ASCEND_PYTHON_EXECUTABLE)
  set(ASCEND_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
add_subdirectory(${KVCACHE_OPS_SOURCE_DIR} ${CMAKE_BINARY_DIR}/kvcache-ops)
install(TARGETS cache_kernels LIBRARY DESTINATION lib)
""".lstrip(),
        encoding="utf-8",
    )
    return wrapper_path


def hipify_wrapper() -> None:
    # Third Party
    from torch.utils.hipify.hipify_python import hipify

    print("Hipifying sources ")

    # Get absolute path for all source files.
    extra_files = [
        os.path.abspath(os.path.join(HIPIFY_DIR, item))
        for item in os.listdir(HIPIFY_DIR)
        if os.path.isfile(os.path.join(HIPIFY_DIR, item))
    ]

    hipify_result = hipify(
        project_directory=HIPIFY_DIR,
        output_directory=HIPIFY_OUT_DIR,
        header_include_dirs=[],
        includes=[],
        extra_files=extra_files,
        show_detailed=True,
        is_pytorch_extension=True,
        hipify_extra_files_only=True,
    )
    hipified_sources = []
    for source in extra_files:
        s_abs = os.path.abspath(source)
        hipified_s_abs = (
            hipify_result[s_abs].hipified_path
            if (
                s_abs in hipify_result
                and hipify_result[s_abs].hipified_path is not None
            )
            else s_abs
        )
        hipified_sources.append(hipified_s_abs)

    assert len(hipified_sources) == len(extra_files)


def cuda_extension() -> tuple[list, dict]:
    # Third Party
    from torch.utils import cpp_extension

    print("Building CUDA extensions")
    global ENABLE_CXX11_ABI
    if ENABLE_CXX11_ABI:
        flag_cxx_abi = "-D_GLIBCXX_USE_CXX11_ABI=1"
    else:
        flag_cxx_abi = "-D_GLIBCXX_USE_CXX11_ABI=0"

    cuda_sources = [
        "csrc/pybind.cpp",
        "csrc/mem_kernels.cu",
        "csrc/cal_cdf.cu",
        "csrc/ac_enc.cu",
        "csrc/ac_dec.cu",
        "csrc/pos_kernels.cu",
        "csrc/mem_alloc.cpp",
        "csrc/utils.cpp",
    ]
    storage_manager_sources = [
        "csrc/storage_manager/bitmap.cpp",
        "csrc/storage_manager/pybind.cpp",
        "csrc/storage_manager/ttl_lock.cpp",
        "csrc/storage_manager/utils.cpp",
    ]
    redis_sources = [
        "csrc/redis/pybind.cpp",
        "csrc/redis/resp.cpp",
    ]
    ext_modules = [
        cpp_extension.CUDAExtension(
            "lmcache.c_ops",
            sources=cuda_sources,
            extra_compile_args={
                "cxx": [flag_cxx_abi, "-std=c++17"],
                "nvcc": [flag_cxx_abi],
            },
        ),
        cpp_extension.CppExtension(
            "lmcache.native_storage_ops",
            sources=storage_manager_sources,
            include_dirs=["csrc/storage_manager"],
            extra_compile_args={
                "cxx": [flag_cxx_abi, "-O3", "-std=c++17"],
            },
        ),
        cpp_extension.CppExtension(
            "lmcache.lmcache_redis",
            sources=redis_sources,
            include_dirs=["csrc/redis"],
            extra_compile_args={
                "cxx": [flag_cxx_abi, "-O3", "-std=c++17"],
            },
        ),
    ]
    cmdclass = {"build_ext": cpp_extension.BuildExtension}
    return ext_modules, cmdclass


def rocm_extension() -> tuple[list, dict]:
    # Third Party
    from torch.utils import cpp_extension

    print("Building ROCM extensions")
    hipify_wrapper()
    hip_sources = [
        "csrc/pybind_hip.cpp",
        "csrc/mem_kernels.hip",
        "csrc/cal_cdf.hip",
        "csrc/ac_enc.hip",
        "csrc/ac_dec.hip",
        "csrc/pos_kernels.hip",
        "csrc/mem_alloc_hip.cpp",
        "csrc/utils_hip.cpp",
    ]
    storage_manager_sources = [
        "csrc/storage_manager/bitmap.cpp",
        "csrc/storage_manager/pybind.cpp",
        "csrc/storage_manager/ttl_lock.cpp",
        "csrc/storage_manager/utils.cpp",
    ]
    redis_sources = [
        "csrc/redis/pybind.cpp",
        "csrc/redis/resp.cpp",
    ]
    define_macros = [("__HIP_PLATFORM_HCC__", "1"), ("USE_ROCM", "1")]
    ext_modules = [
        cpp_extension.CppExtension(
            "lmcache.c_ops",
            sources=hip_sources,
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-std=c++17",
                ],
            },
            include_dirs=[
                os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "include")
            ],
            library_dirs=[
                os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "lib")
            ],
            define_macros=define_macros,
        ),
        cpp_extension.CppExtension(
            "lmcache.native_storage_ops",
            sources=storage_manager_sources,
            include_dirs=["csrc/storage_manager"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
            },
        ),
        cpp_extension.CppExtension(
            "lmcache.lmcache_redis",
            sources=redis_sources,
            include_dirs=["csrc/redis"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
            },
        ),
    ]
    cmdclass = {"build_ext": cpp_extension.BuildExtension}
    return ext_modules, cmdclass


def ascend_extension() -> tuple[list, dict]:
    from torch.utils import cpp_extension

    print("Building Ascend extensions")
    flag_cxx_abi = (
        "-D_GLIBCXX_USE_CXX11_ABI=1" if ENABLE_CXX11_ABI else "-D_GLIBCXX_USE_CXX11_ABI=0"
    )

    ascend_home = _get_ascend_home_path()
    arch_subdir = _ascend_arch_subdir()
    torch_npu_path = _get_torch_npu_path()
    kvcache_source_root = ROOT_DIR / "third_party" / "kvcache-ops"
    if not kvcache_source_root.exists():
        raise RuntimeError(
            "BUILD_WITH_ASCEND=1 requires third_party/kvcache-ops in the materialized workspace."
        )

    kvcache_build_root = ROOT_DIR / "build" / "ascend-kvcache-ops"
    kvcache_install_root = kvcache_build_root / "install"
    kvcache_lib_root = kvcache_install_root / "lib"

    include_dirs = [
        str(ROOT_DIR / "csrc" / "ascend"),
        str(ROOT_DIR / "csrc" / "ascend" / "common"),
        str(ROOT_DIR / "third_party" / "kvcache-ops"),
        str(ROOT_DIR / "third_party" / "kvcache-ops" / "kernels"),
        str(ROOT_DIR / "third_party" / "kvcache-ops" / "kvcache-ops"),
        str(ROOT_DIR / "third_party" / "kvcache-ops" / "kvcache-ops" / "kernels"),
        str(torch_npu_path / "include"),
        str(ascend_home / "include"),
        str(ascend_home / arch_subdir / "ascendc" / "include"),
        str(ascend_home / arch_subdir / "include" / "experiment" / "platform"),
    ]
    library_dirs = [
        str(kvcache_lib_root),
        str(torch_npu_path / "lib"),
        str(ascend_home / "lib64"),
        str(ascend_home / arch_subdir / "devlib"),
    ]
    runtime_rpaths = [
        "$ORIGIN",
        str(kvcache_lib_root),
        str(torch_npu_path / "lib"),
        str(ascend_home / "lib64"),
        str(ascend_home / arch_subdir / "devlib"),
    ]
    extra_link_args = [f"-Wl,-rpath,{path}" for path in runtime_rpaths]

    storage_manager_sources = [
        "csrc/storage_manager/bitmap.cpp",
        "csrc/storage_manager/pybind.cpp",
        "csrc/storage_manager/ttl_lock.cpp",
        "csrc/storage_manager/utils.cpp",
    ]
    redis_sources = [
        "csrc/redis/pybind.cpp",
        "csrc/redis/resp.cpp",
    ]

    class AscendBuildExtension(cpp_extension.BuildExtension):
        def _build_kvcache_ops(self) -> None:
            wrapper_root = ROOT_DIR / "build" / "ascend-kvcache-wrapper"
            wrapper_cmake = _write_kvcache_ops_wrapper(wrapper_root)
            soc_version = _get_soc_version()
            aicore_arch = _get_aicore_arch_number(ascend_home, soc_version)

            configure_cmd = [
                "cmake",
                "-S",
                str(wrapper_cmake.parent),
                "-B",
                str(kvcache_build_root),
                f"-DKVCACHE_OPS_SOURCE_DIR={kvcache_source_root}",
                f"-DASCEND_CANN_PACKAGE_PATH={ascend_home}",
                f"-DASCEND_PYTHON_EXECUTABLE={sys.executable}",
                f"-DCMAKE_INSTALL_PREFIX={kvcache_install_root}",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DSOC_VERSION={soc_version}",
            ]
            if aicore_arch:
                configure_cmd.append(f"-DASCEND_AICORE_ARCH={aicore_arch}")

            subprocess.check_call(configure_cmd, cwd=ROOT_DIR)
            subprocess.check_call(
                ["cmake", "--build", str(kvcache_build_root), "-j"],
                cwd=ROOT_DIR,
            )
            subprocess.check_call(
                ["cmake", "--install", str(kvcache_build_root)],
                cwd=ROOT_DIR,
            )

        def _copy_runtime_artifacts(self) -> None:
            package_dir = Path(self.build_lib) / "lmcache"
            package_dir.mkdir(parents=True, exist_ok=True)
            for pattern in ("libcache_kernels.so", "libcache_kernels.so*"):
                for src in glob.glob(str(kvcache_lib_root / pattern)):
                    dst = package_dir / Path(src).name
                    if Path(src).resolve() == dst.resolve():
                        continue
                    shutil.copy2(src, dst)

        def run(self):
            self._build_kvcache_ops()
            super().run()
            self._copy_runtime_artifacts()

    ext_modules = [
        cpp_extension.CppExtension(
            "lmcache.c_ops",
            sources=[
                "csrc/ascend/pybind.cpp",
                "csrc/ascend/mem_kernels.cpp",
                "csrc/ascend/mem_alloc.cpp",
                "csrc/ascend/utils.cpp",
                "csrc/ascend/common/dcmi_management.cpp",
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=[
                "cache_kernels",
                "torch_npu",
                "ascendcl",
                "platform",
                "tiling_api",
                "numa",
                "dl",
            ],
            extra_compile_args={
                "cxx": [flag_cxx_abi, "-O3", "-std=c++17"],
            },
            extra_link_args=extra_link_args,
        ),
        cpp_extension.CppExtension(
            "lmcache.native_storage_ops",
            sources=storage_manager_sources,
            include_dirs=["csrc/storage_manager"],
            extra_compile_args={
                "cxx": [flag_cxx_abi, "-O3", "-std=c++17"],
            },
        ),
        cpp_extension.CppExtension(
            "lmcache.lmcache_redis",
            sources=redis_sources,
            include_dirs=["csrc/redis"],
            extra_compile_args={
                "cxx": [flag_cxx_abi, "-O3", "-std=c++17"],
            },
        ),
    ]
    cmdclass = {"build_ext": AscendBuildExtension}
    return ext_modules, cmdclass


def source_dist_extension() -> tuple[list, dict]:
    print("Not building CUDA/HIP extensions for sdist")
    return [], {}


if __name__ == "__main__":
    if BUILDING_SDIST:
        get_extension = source_dist_extension
    elif BUILD_WITH_HIP:
        get_extension = rocm_extension
    elif BUILD_WITH_ASCEND:
        get_extension = ascend_extension
    else:
        get_extension = cuda_extension

    ext_modules, cmdclass = get_extension()

    setup(
        packages=find_packages(
            exclude=("csrc",)
        ),
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        include_package_data=True,
        package_data={"lmcache": ["*.so"]},
    )