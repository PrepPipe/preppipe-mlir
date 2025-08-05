# Disables generation of "version soname" (i.e. libFoo.so.<version>), which
# causes pure duplication as part of Python wheels.
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

# The directory at which the Python import tree begins.
# See documentation for `declare_mlir_python_sources`'s ROOT_DIR
# argument.
set(PREPPIPE_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/preppipe_mlir")

# We vendor our own MLIR instance in the `preppipe_mlir` namespace.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=preppipe_mlir.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(PrepPipeMLIRPythonSources)
declare_mlir_python_sources(PrepPipeMLIRPythonExtensions)

declare_mlir_python_sources(PrepPipeMLIRPythonSources.Dialects
  ROOT_DIR "${PREPPIPE_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT PrepPipeMLIRPythonSources
)

declare_mlir_python_sources(PrepPipeMLIRPythonSources.Core
  ROOT_DIR "${PREPPIPE_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT PrepPipeMLIRPythonSources
)

declare_mlir_python_sources(PrepPipeMLIRPythonSources.Core.Python
  ROOT_DIR "${PREPPIPE_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT PrepPipeMLIRPythonSources.Core
  SOURCES
    helper.py
)

declare_mlir_python_sources(PrepPipeMLIRSiteInitialize
  ROOT_DIR "${PREPPIPE_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT PrepPipeMLIRPythonSources
  SOURCES
    _mlir_libs/_site_initialize_0.py
)

################################################################################
# Dialect bindings
################################################################################

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT PrepPipeMLIRPythonSources.Dialects
  ROOT_DIR "${PREPPIPE_MLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/PrepPipeOps.td
  SOURCES_GLOB dialects/preppipe.py
  DIALECT_NAME preppipe
  GEN_ENUM_BINDINGS_TD_FILE
    "dialects/PrepPipeAttrs.td"
)

################################################################################
# Extensions
################################################################################

set(PYTHON_SOURCE_DIR "${PREPPIPE_MLIR_SOURCE_DIR}/lib/Bindings/Python")

declare_mlir_python_extension(PrepPipeMLIRPythonExtensions.Main
  MODULE_NAME _prepPipeMlirRegisterEverything
  ADD_TO_PARENT PrepPipeMLIRPythonExtensions
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    RegisterEverything.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    PrepPipeMLIRCAPIRegisterEverything
    MLIRCAPIPrepPipe
)

declare_mlir_python_extension(PrepPipeMLIRPythonExtensions.Dialects.PrepPipe.Pybind
  MODULE_NAME _prepPipeMlirDialectsPrepPipe
  ADD_TO_PARENT PrepPipeMLIRPythonSources.Dialects.preppipe
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectPrepPipe.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIPrepPipe
)

declare_mlir_python_extension(PrepPipeMLIRPythonExtensions.PrepPipeDialectPasses
  MODULE_NAME _prepPipeMlirPrepPipePasses
  ADD_TO_PARENT PrepPipeMLIRPythonSources.Dialects.preppipe
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    PrepPipePasses.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIPrepPipe
)

################################################################################
# Generate packages and shared library
# Downstreams typically will not use these, but they are useful for local
# testing.
################################################################################

set(_source_components
  # TODO: Core is now implicitly building/registering all dialects, increasing
  # build burden by ~5x. Make it stop.
  # TODO: Reduce dependencies.
  MLIRPythonSources
  MLIRPythonExtension.Core
  MLIRPythonExtension.RegisterEverything
  PrepPipeMLIRPythonSources
  PrepPipeMLIRPythonExtensions
  PrepPipeMLIRSiteInitialize
)

add_mlir_python_common_capi_library(PrepPipeMLIRAggregateCAPI
  INSTALL_COMPONENT PrepPipeMLIRPythonModules
  INSTALL_DESTINATION python_packages/preppipe_mlir/preppipe_mlir/_mlir_libs
  OUTPUT_DIRECTORY "${PREPPIPE_MLIR_PYTHON_PACKAGES_DIR}/preppipe_mlir/preppipe_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT ".."
  DECLARED_SOURCES ${_source_components}
)

add_mlir_python_modules(PrepPipeMLIRPythonModules
  ROOT_PREFIX "${PREPPIPE_MLIR_PYTHON_PACKAGES_DIR}/preppipe_mlir/preppipe_mlir"
  INSTALL_PREFIX "python_packages/preppipe_mlir/preppipe_mlir"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    PrepPipeMLIRAggregateCAPI
)

# add_dependencies(PrepPipeMLIRPythonModules preppipe-mlir-opt)
