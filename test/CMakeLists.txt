llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# Passed to lit.site.cfg.py.in to set up the path where to find libraries.
set(MLIR_LIB_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(PREPPIPE_MLIR_TEST_DEPENDS
  FileCheck count not split-file
  #PrepPipeMLIRPythonModules
  preppipe-mlir-opt
)

add_lit_testsuite(check-preppipe-mlir "Running the preppipe-mlir regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${PREPPIPE_MLIR_TEST_DEPENDS}
)

set_target_properties(check-preppipe-mlir PROPERTIES FOLDER "Tests")

add_lit_testsuites(PREPPIPE_MLIR ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${PREPPIPE_MLIR_TEST_DEPENDS})

#if (MLIR_ENABLE_BINDINGS_PYTHON)
#  add_subdirectory(python)
#endif()
