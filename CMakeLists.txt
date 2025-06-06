cmake_minimum_required(VERSION 3.15)
project(polyframe LANGUAGES CXX)

# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CXX_FLAGS_DEBUG "-g" CACHE STRING "Debug compiler flags" FORCE)
# set(CMAKE_C_FLAGS_DEBUG "-g" CACHE STRING "Debug compiler flags" FORCE)

# -----------------------------------------------------------------------------
# Locate Python + NumPy
# -----------------------------------------------------------------------------
find_package(Python COMPONENTS Interpreter NumPy REQUIRED)

# -----------------------------------------------------------------------------
# Locate nanobind
# -----------------------------------------------------------------------------
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${nanobind_ROOT}")
find_package(nanobind CONFIG REQUIRED)

# -----------------------------------------------------------------------------
# Configure the Python extension
# -----------------------------------------------------------------------------
set(MODULE_NAME _polyframe)
nanobind_add_module(${MODULE_NAME}
    src/polyframe/bindings.cpp
)

# -----------------------------------------------------------------------------
# Include directories
# -----------------------------------------------------------------------------
target_include_directories(${MODULE_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/Eigen
    ${Python_NumPy_INCLUDE_DIR}
)

# -----------------------------------------------------------------------------
# Installation & stub generation
# -----------------------------------------------------------------------------
install(TARGETS ${MODULE_NAME}
        RUNTIME DESTINATION ${SKBUILD_PROJECT_NAME}
        LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME}
        ARCHIVE DESTINATION ${SKBUILD_PROJECT_NAME})

# --- ensure on Windows the .pyd has no ABI‐tag suffix
if (WIN32)
  set_target_properties(${MODULE_NAME} PROPERTIES
    OUTPUT_NAME "${MODULE_NAME}"
    SUFFIX ".pyd"                     # force the .pyd extension
  )
endif()

# --- regenerate stub with an explicit import path
add_custom_command(TARGET ${MODULE_NAME} POST_BUILD
    COMMAND "${Python_EXECUTABLE}" -m nanobind.stubgen
            -i "$<TARGET_FILE_DIR:${MODULE_NAME}>"   # <-- search path
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
