function(EnableDept)
    cmake_parse_arguments(DEPT "" "NAME;TAG" "GIT_URLS" ${ARGN})
    find_program(GIT_EXECUTABLE git)
    if(NOT GIT_EXECUTABLE)
        message(FATAL_ERROR "git not found!")
    endif()
    if(NOT DEFINED VLLM_SPARSE_VENDOR_VALIDATE_URL)
        set(VLLM_SPARSE_VENDOR_VALIDATE_URL ON)
    endif()

    set(VALID_GIT_URL "")
    if(VLLM_SPARSE_VENDOR_VALIDATE_URL)
        foreach(GIT_URL IN LISTS DEPT_GIT_URLS)
            execute_process(
                COMMAND ${GIT_EXECUTABLE} ls-remote --heads "${GIT_URL}"
                RESULT_VARIABLE GIT_RESULT
                OUTPUT_QUIET
                ERROR_QUIET
                TIMEOUT 10
            )
            if(GIT_RESULT EQUAL 0)
                set(VALID_GIT_URL ${GIT_URL})
                break()
            endif()
        endforeach()
        if(NOT VALID_GIT_URL)
            message(FATAL_ERROR "all urls for ${DEPT_NAME} are not reachable!")
        endif()
    else()
        list(GET DEPT_GIT_URLS 0 VALID_GIT_URL)
    endif()

    message(STATUS "Fetching ${DEPT_NAME}(${DEPT_TAG}) from ${VALID_GIT_URL}")
    FetchContent_Declare(
        ${DEPT_NAME}
        GIT_REPOSITORY ${VALID_GIT_URL}
        GIT_TAG ${DEPT_TAG}
        GIT_SHALLOW TRUE
        GIT_PROGRESS FALSE
    )
    string(TOUPPER ${DEPT_NAME} NAME_UPPER)
    set(${NAME_UPPER}_INSTALL OFF CACHE INTERNAL "" FORCE)
    set(${NAME_UPPER}_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
    set(${NAME_UPPER}_BUILD_EXAMPLES OFF CACHE INTERNAL "" FORCE)
    FetchContent_MakeAvailable(${DEPT_NAME})
endfunction()

if(DOWNLOAD_DEPENDENCE)
    include(FetchContent)
    set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
    EnableDept(
        NAME fmt
        TAG 11.2.0
        GIT_URLS
            https://github.com/fmtlib/fmt.git
            https://git.codehub.xfusion.com/OpenSourceCenter/fmt.git
	    ssh://git@git.codehub.xfusion.com:2222/OpenSourceCenter/fmt.git
    )
    EnableDept(
        NAME spdlog
        TAG v1.9.2
        GIT_URLS
            https://github.com/gabime/spdlog.git
            https://git.codehub.xfusion.com/OpenSourceCenter/spdlog.git
	    ssh://git@git.codehub.xfusion.com:2222/OpenSourceCenter/spdlog.git
    )
    EnableDept(
        NAME pybind11
        TAG v2.13.6
        GIT_URLS
            https://github.com/pybind/pybind11.git
            https://git.codehub.xfusion.com/OpenSourceCenter/pybind11.git
	    ssh://git@git.codehub.xfusion.com:2222/OpenSourceCenter/pybind11.git
    )
    FetchContent_MakeAvailable(pybind11)
else()
    find_package(fmt CONFIG REQUIRED)
    find_package(spdlog CONFIG REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    message(STATUS "pybind11 found, checking targets...")
    if(TARGET pybind11::headers)
        message(STATUS "  Found pybind11::headers")
    else()
        message(STATUS "  pybind11::headers NOT found")
    endif()
    if(TARGET pybind11::pybind11_headers)
        message(STATUS "  Found pybind11::pybind11_headers")
    else()
        message(STATUS "  pybind11::pybind11_headers NOT found")
    endif()
    if(TARGET pybind11::pybind11)
        message(STATUS "  Found pybind11::pybind11")
    else()
        message(STATUS "  pybind11::pybind11 NOT found")
    endif()
endif()

if(NOT TARGET fmt)
    if(TARGET fmt::fmt)
        add_library(fmt ALIAS fmt::fmt)
    elseif(TARGET fmt::fmt-header-only)
        add_library(fmt ALIAS fmt::fmt-header-only)
    endif()
endif()

if(TARGET spdlog::spdlog AND NOT TARGET spdlog)
    add_library(spdlog ALIAS spdlog::spdlog)
endif()

if(NOT TARGET pybind11::headers)
    if(TARGET pybind11::pybind11_headers)
        add_library(pybind11::headers ALIAS pybind11::pybind11_headers)
    elseif(TARGET pybind11::pybind11)
        # Some pybind11 versions may only have pybind11::pybind11
        add_library(pybind11::headers INTERFACE)
        target_link_libraries(pybind11::headers INTERFACE pybind11::pybind11)
    else()
        message(WARNING "pybind11::headers target not found, trying to find alternative")
    endif()
endif()
