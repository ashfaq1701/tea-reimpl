cmake_minimum_required(VERSION 3.18)
project(tea_reimpl LANGUAGES CXX)

# --- Standard / flags -------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# default to Release if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo)
endif()

# project-wide flags. Add per-target if a single target needs different.
add_library(tea_compile_flags INTERFACE)
target_compile_options(tea_compile_flags INTERFACE
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>
    $<$<CONFIG:RelWithDebInfo>:-O3 -march=native -g>
    $<$<CONFIG:Debug>:-O0 -g>)

# --- OpenMP -----------------------------------------------------------------
find_package(OpenMP REQUIRED)

# --- Third-party deps via FetchContent --------------------------------------
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

option(TEA_BUILD_TESTS "Build unit tests" ON)

if(TEA_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    # silence googletest's own warnings + suppress its install/uninstall targets
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK   OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# absl::flat_hash_set is needed by Phase 5 (Node2Vec neighbor sets) — added
# at that point via FetchContent. Not pulled in here to keep Phase 0 fast.

# --- The tea library target -------------------------------------------------
# Library target exists from Phase 0 so downstream targets can link. Sources
# get added as phases land.
add_library(tea STATIC
    src/lib_marker.cpp
    src/graph.cpp)
target_include_directories(tea PUBLIC include)
target_link_libraries(tea
    PUBLIC  tea_compile_flags
    PUBLIC  OpenMP::OpenMP_CXX)

# --- CLI binary -------------------------------------------------------------
add_executable(tea_walk cli/tea_walk.cpp)
target_link_libraries(tea_walk PRIVATE tea tea_compile_flags)

# --- Tests ------------------------------------------------------------------
if(TEA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# --- Diagnostic banner -----------------------------------------------------
message(STATUS "------------------------------------------------------------")
message(STATUS "tea_reimpl configured")
message(STATUS "  build type     : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler   : ${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
message(STATUS "  OpenMP version : ${OpenMP_CXX_VERSION}")
message(STATUS "  build tests    : ${TEA_BUILD_TESTS}")
message(STATUS "------------------------------------------------------------")
