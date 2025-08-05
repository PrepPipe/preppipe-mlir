add_mlir_library(MLIRPrepPipeDialect
  PrepPipe.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/preppipe-mlir/Dialect/PrepPipe

  DEPENDS
  MLIRPrepPipeOpsIncGen
  MLIRPrepPipeTypesIncGen

  LINK_LIBS PUBLIC
  MLIRBytecodeOpInterface
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRIR
  MLIRSupport
)

preppipe_mlir_target_includes(MLIRPrepPipeDialect)
