cmake_minimum_required(VERSION 3.26)
project(vllm_sparse_native LANGUAGES CXX)
project(vllm_sparse_native LANGUAGES CXX CUDA)  # 添加 CUDA


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

add_subdirectory(kvstore)
add_subdirectory(offload_ops)
add_subdirectory(prefetch_engine)
add_subdirectory(paged_kmeans)
