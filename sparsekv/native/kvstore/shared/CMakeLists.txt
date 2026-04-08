add_subdirectory(vendor)
add_subdirectory(infra)

# storeinfra: native/kvstore 的“共享基础设施库”
#
# - 负责把 shared/file 的实现编译进一个可复用的 STATIC 库；
# - 通过链接 shared/infra 下的一组 OBJECT libs（infra_*）把日志、状态码、线程等能力带进来；
# - 由上层模块（core/local_kvstore）按需链接，从而避免重复编译。
add_library(storeinfra STATIC)
target_include_directories(storeinfra PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/file
)
file(GLOB_RECURSE VLLM_SPARSE_STOREINFRA_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/file/*.cc"
)
target_sources(storeinfra PRIVATE ${VLLM_SPARSE_STOREINFRA_SOURCES})
target_link_libraries(storeinfra PUBLIC
    infra_status
    infra_logger
    infra_template
    infra_thread
    infra_time
)

# trans: 通用传输抽象，当前 _local_kvstore 不强依赖
# 默认关闭以缩短构建链路；需要时可在配置阶段开启。
option(VLLM_SPARSE_KVSTORE_BUILD_TRANS "Build kvstore shared/trans" OFF)
if(VLLM_SPARSE_KVSTORE_BUILD_TRANS)
    add_subdirectory(trans)
endif()
