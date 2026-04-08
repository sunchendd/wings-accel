add_library(infra_status INTERFACE)
target_include_directories(infra_status INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<$<TARGET_EXISTS:fmt>:$<TARGET_PROPERTY:fmt,INTERFACE_INCLUDE_DIRECTORIES>>
)
target_link_libraries(infra_status INTERFACE fmt)

file(GLOB UCMINFRA_LOGGER_SOURCE_FILES "logger/*.*")
file(GLOB_RECURSE UCMINFRA_LOGGER_DETAIL_SOURCE_FILES "logger/${LOGGER_BACKEND}/*.cc")
add_library(infra_logger OBJECT ${UCMINFRA_LOGGER_SOURCE_FILES} ${UCMINFRA_LOGGER_DETAIL_SOURCE_FILES})
target_include_directories(infra_logger PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<$<TARGET_EXISTS:fmt>:$<TARGET_PROPERTY:fmt,INTERFACE_INCLUDE_DIRECTORIES>>
    $<$<TARGET_EXISTS:spdlog>:$<TARGET_PROPERTY:spdlog,INTERFACE_INCLUDE_DIRECTORIES>>
)
target_link_libraries(infra_logger PUBLIC fmt spdlog)

# file(GLOB_RECURSE UCMINFRA_TEMPLATE_SOURCE_FILES "template/*.*")
# add_library(infra_template OBJECT ${UCMINFRA_TEMPLATE_SOURCE_FILES})
# target_include_directories(infra_template PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# file(GLOB_RECURSE UCMINFRA_THREAD_SOURCE_FILES "thread/*.*")
# add_library(infra_thread OBJECT ${UCMINFRA_THREAD_SOURCE_FILES})
# target_include_directories(infra_thread PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# file(GLOB_RECURSE UCMINFRA_TIME_SOURCE_FILES "time/*.*")
# add_library(infra_time OBJECT ${UCMINFRA_TIME_SOURCE_FILES})
# target_include_directories(infra_time PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(infra_template INTERFACE)
target_include_directories(infra_template INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(infra_thread INTERFACE)
target_include_directories(infra_thread INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(infra_time INTERFACE)
target_include_directories(infra_time INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
