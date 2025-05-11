llvm_canonicalize_cmake_booleans(
    MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# if FILECHECK_EXECUTABLE is not manually set, try to find one
if(NOT EXISTS ${FILECHECK_EXECUTABLE})
    find_program(FILECHECK_EXECUTABLE NAMES FileCheck PATHS "/usr/local/bin" "/usr/bin" "/opt")
endif()

if(NOT EXISTS ${FILECHECK_EXECUTABLE})
    message(FATAL_ERROR "Filecheck utility could not be found")
endif()

message(STATUS "Filecheck found at : ${FILECHECK_EXECUTABLE}")

execute_process(COMMAND "${FILECHECK_EXECUTABLE}" --version
    RESULT_VARIABLE IS_FILECHECK_VALID
    OUTPUT_VARIABLE FILECHECK_VERSION
    ERROR_VARIABLE FILECHECK_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)

if(NOT DEFINED IS_FILECHECK_VALID)
    message(FATAL_ERROR "Failed to execute FileCheck command")
elseif(NOT "${IS_FILECHECK_VALID}" EQUAL "0")
    message(FATAL_ERROR "FileCheck execution failed with code: ${IS_FILECHECK_VALID}")
endif()

if(FILECHECK_VERSION)
    message(STATUS "FileCheck version: ${FILECHECK_VERSION}")
endif()

set(MLIR_TEMPLATE_TEST_DEPENDS
    mlir-template-opt
    mlir-template-translate
)

# TODO : enable this after configuring python bindings
# if(MLIR_ENABLE_BINDINGS_PYTHON)
#   list(APPEND MLIR_TEMPLATE_TEST_DEPENDS StandalonePythonModules)
# endif()

add_lit_testsuite(check-mlir-template "Running the standalone regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${MLIR_TEMPLATE_TEST_DEPENDS}
)
set_target_properties(check-mlir-template PROPERTIES FOLDER "Tests")

add_lit_testsuites(MLIR_TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${MLIR_TEMPLATE_TEST_DEPENDS})
