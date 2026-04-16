add_library(localstore STATIC
    src/localstore.cc
)
target_include_directories(localstore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../core
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared/infra
)
target_link_libraries(localstore PUBLIC storeinfra storedevice)

pybind11_add_module(_local_kvstore
    bindings.cpp
)
target_include_directories(_local_kvstore PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../core
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared/infra
)
target_link_libraries(_local_kvstore PRIVATE localstore)
set_target_properties(_local_kvstore PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${VLLM_SPARSE_KVSTORE_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${VLLM_SPARSE_KVSTORE_OUTPUT_DIR}
)
