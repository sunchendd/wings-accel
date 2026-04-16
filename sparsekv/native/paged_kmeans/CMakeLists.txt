if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.18)
    project(_paged_kmeans LANGUAGES CXX CUDA)
endif()

# =========================
# Build switches / defaults
# =========================

# This module only targets CUDA right now (CUTLASS Sm80 kernel).
if(NOT DEFINED RUNTIME_ENVIRONMENT)
    set(RUNTIME_ENVIRONMENT "cuda")
endif()
if(NOT RUNTIME_ENVIRONMENT STREQUAL "cuda")
    message(FATAL_ERROR "Only RUNTIME_ENVIRONMENT=cuda is supported, got: ${RUNTIME_ENVIRONMENT}")
endif()

if(NOT CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()

# Default to a small set of common architectures.
# You can override via: -DCMAKE_CUDA_ARCHITECTURES="80" (or "80;86;89;90", etc.)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
endif()

# Allow overriding where the produced .so will be placed.
# Default output: vllm/v1/sparse/native (centralized native extension package).
set(PAGED_KMEANS_OUTPUT_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    CACHE PATH
    "Output directory for the built Python extension (.so)"
)

# =========================
# Toolchain / dependencies
# =========================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_RESULT
)
if(NOT pybind11_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to find pybind11 cmake files. Try: pip install pybind11")
endif()
find_package(pybind11 CONFIG REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import os, torch; print(os.path.dirname(os.path.abspath(torch.__file__)))"
    OUTPUT_VARIABLE PYTORCH_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYTORCH_RESULT
)
if(NOT PYTORCH_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to find PyTorch installation")
endif()

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(int(getattr(torch._C, '_GLIBCXX_USE_CXX11_ABI', 1)))"
    OUTPUT_VARIABLE CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE ABI_RESULT
)
if(NOT ABI_RESULT EQUAL 0)
    set(CXX11_ABI "1")
endif()

find_package(CUDAToolkit REQUIRED)

set(CUTLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/cutlass)

# =========================
# Extension target
# =========================

set(INCLUDE_DIRS
    ${PYTORCH_PATH}/include/torch/csrc/api/include
    ${PYTORCH_PATH}/include
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/examples/common
    ${CUTLASS_DIR}/tools/util/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set(LIBRARY_DIRS
    ${PYTORCH_PATH}/lib
)

set(LIBRARIES
    torch
    c10
    torch_cpu
    torch_python
    torch_cuda
    c10_cuda
)

set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/batch_gemm_softmax.cu
)

pybind11_add_module(_paged_kmeans ${SOURCES})

set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/batch_gemm_softmax.cu
    PROPERTIES
    LANGUAGE CUDA
)

target_compile_definitions(_paged_kmeans PRIVATE
    TORCH_EXTENSION_NAME=_paged_kmeans
    _GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}
)

# Some environments require explicitly forcing C++17 on the command line.
target_compile_features(_paged_kmeans PRIVATE cxx_std_17)

target_include_directories(_paged_kmeans PRIVATE ${INCLUDE_DIRS})
target_link_directories(_paged_kmeans PRIVATE ${LIBRARY_DIRS})

target_compile_options(_paged_kmeans PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-std=c++17>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-std=c++17;--expt-relaxed-constexpr>
)

target_link_libraries(_paged_kmeans PRIVATE
    ${LIBRARIES}
    CUDA::cudart
)

if(TARGET CUDA::cuda_driver)
    target_link_libraries(_paged_kmeans PRIVATE CUDA::cuda_driver)
endif()

set_target_properties(_paged_kmeans PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PAGED_KMEANS_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${PAGED_KMEANS_OUTPUT_DIR}
)
