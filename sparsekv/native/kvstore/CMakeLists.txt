if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.18)
    project(vllm_sparse_kvstore LANGUAGES CXX)
endif()

if(NOT DEFINED RUNTIME_ENVIRONMENT)
    set(RUNTIME_ENVIRONMENT "cuda")
endif()

if(NOT DEFINED LOGGER_BACKEND)
    set(LOGGER_BACKEND "spdlog")
endif()

if(NOT DEFINED DOWNLOAD_DEPENDENCE)
    set(DOWNLOAD_DEPENDENCE ON)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

if(RUNTIME_ENVIRONMENT STREQUAL "cuda")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
elseif(RUNTIME_ENVIRONMENT STREQUAL "maca")
    if(NOT DEFINED CUDA_ROOT)
        message(FATAL_ERROR "CUDA_ROOT must be set when RUNTIME_ENVIRONMENT=maca.")
    endif()
    set(CMAKE_CUDA_COMPILER ${CUDA_ROOT}/bin/cucc)
    list(APPEND CMAKE_MODULE_PATH "${CUDA_ROOT}/cmake_module/maca")
    enable_language(CUDA)
endif()

if(NOT DEFINED VLLM_SPARSE_KVSTORE_OUTPUT_DIR)
    set(VLLM_SPARSE_KVSTORE_OUTPUT_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
        CACHE PATH
        "Output directory for the built Python extension (.so)"
    )
endif()

add_subdirectory(shared)
find_package(pybind11 CONFIG REQUIRED)
add_subdirectory(core)
add_subdirectory(local_kvstore)
