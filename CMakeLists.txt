cmake_minimum_required(VERSION 3.15)
project(polyframe LANGUAGES CXX)

# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CXX_FLAGS_DEBUG "-g" CACHE STRING "Debug compiler flags" FORCE)
# set(CMAKE_C_FLAGS_DEBUG "-g" CACHE STRING "Debug compiler flags" FORCE)
# set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native -DNDEBUG")


# -----------------------------------------------------------------------------
# 1) Vendor Eigen via FetchContent
# -----------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
)
set(BUILD_TESTING OFF CACHE BOOL \"Disable Eigen’s tests\")
FetchContent_MakeAvailable(eigen)

# -----------------------------------------------------------------------------
# 2) Locate Python + NumPy
# -----------------------------------------------------------------------------
find_package(Python COMPONENTS Interpreter NumPy REQUIRED)

# -----------------------------------------------------------------------------
# 3) Locate nanobind
# -----------------------------------------------------------------------------
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${nanobind_ROOT}")
find_package(nanobind CONFIG REQUIRED)

# -----------------------------------------------------------------------------
# 4) Configure the Python extension
# -----------------------------------------------------------------------------
set(MODULE_NAME _polyframe)
nanobind_add_module(${MODULE_NAME}
    src/polyframe/bindings.cpp
)

# -----------------------------------------------------------------------------
# 5) Include directories
# -----------------------------------------------------------------------------
target_include_directories(${MODULE_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    ${Python_NumPy_INCLUDE_DIR}
)

# Eigen is now available via FetchContent; use its target
# nanobind provided the nanobind_CMAKE_DIR and target

# -----------------------------------------------------------------------------
# 6) Link libraries
# -----------------------------------------------------------------------------
target_link_libraries(${MODULE_NAME} PRIVATE
    Eigen3::Eigen
)

# -----------------------------------------------------------------------------
# 7) Installation & stub generation
# -----------------------------------------------------------------------------
install(TARGETS ${MODULE_NAME}
        RUNTIME DESTINATION ${SKBUILD_PROJECT_NAME}
        LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME}
        ARCHIVE DESTINATION ${SKBUILD_PROJECT_NAME})

add_custom_command(TARGET ${MODULE_NAME} POST_BUILD
    COMMAND "${Python_EXECUTABLE}" -m nanobind.stubgen
            -m ${MODULE_NAME}
            -o "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}.pyi"
    COMMENT "Generating PEP-561 stub for ${MODULE_NAME}"
)
add_custom_command(TARGET ${MODULE_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "from typing import Union, Sequence" > "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}.tmp.pyi"
    COMMAND ${CMAKE_COMMAND} -E cat "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}.pyi" >> "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}.tmp.pyi"
    COMMAND ${CMAKE_COMMAND} -E rename "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}.tmp.pyi" "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}.pyi"
    COMMENT "Injecting typing imports into ${MODULE_NAME}.pyi"
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}.pyi"
        DESTINATION ${PROJECT_NAME})

# -----------------------------------------------------------------------------
# EOF
# -----------------------------------------------------------------------------
