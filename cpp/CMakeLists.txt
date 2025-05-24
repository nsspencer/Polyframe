cmake_minimum_required(VERSION 3.15)
project(polyframe)

# Tell CMake to require C++17
set(CMAKE_CXX_STANDARD        17)       # Use the C++17 language standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)     # Error out if C++17 is unavailable
set(CMAKE_CXX_EXTENSIONS     OFF)       # Don’t use compiler-specific extensions

# Default to Release and turn on high-level optimizations
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Release flags (enable optimizations, disable asserts)
set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -DNDEBUG -ffast-math -fvisibility=hidden"
)

# Pick the right Python FIND options
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter ${DEV_MODULE})

# — point at your local Eigen clone —
set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../external/eigen")
if (NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Core")
  message(FATAL_ERROR "Eigen headers not found in ${EIGEN3_INCLUDE_DIR}")
endif()

# make sure compiler sees Eigen
include_directories(${EIGEN3_INCLUDE_DIR})

# default to Release if no build type is specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    Debug Release MinSizeRel RelWithDebInfo
  )
endif()

# locate nanobind (from pip/conda)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE NANOBIND_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_ROOT}")
find_package(nanobind CONFIG REQUIRED)

set(NUMPY_INCLUDE_DIR "/opt/conda/lib/python3.12/site-packages/numpy/_core/include")

# print the NUMPY_INCLUDE_DIR to the console
message(STATUS "NumPy include directory: ${NUMPY_INCLUDE_DIR}")


# build the Python extension: only list your .cpp files here
nanobind_add_module(
  polyframe         # extension name
  bindings.cpp            # your source
)

# (a) Whole‑program optimisation – removes indirect calls inside nanobind
set_target_properties(polyframe PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

# (b) Nanobind’s micro‑ops for tuples/lists
target_compile_definitions(polyframe PRIVATE NB_OPT_FAST_OPS)

# optionally, tighten the include path to just this target
target_include_directories(polyframe PRIVATE ${EIGEN3_INCLUDE_DIR})

target_include_directories(polyframe PRIVATE
    ${NUMPY_INCLUDE_DIR}
)


# Make sure our Release flags get applied to the module target, too
target_compile_options(polyframe PRIVATE
  $<$<CONFIG:Release>:-O3;-DNDEBUG>
)

# ─────── POST-BUILD: copy the .so/.pyd back to the source dir ───────
add_custom_command(TARGET polyframe
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                           $<TARGET_FILE:polyframe>
                           ${CMAKE_SOURCE_DIR}/$<TARGET_FILE_NAME:polyframe>
                   COMMENT "Copying polyframe to source directory"
)