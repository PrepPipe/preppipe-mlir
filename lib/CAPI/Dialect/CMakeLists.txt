add_mlir_public_c_api_library(MLIRCAPIPrepPipe
  PrepPipe.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/preppipe-mlir-c/

  ENABLE_AGGREGATION

  PARTIAL_SOURCES_INTENDED
  # DEPENDS
  # MLIRPREPPIPETransformsIncGen

  LINK_LIBS PUBLIC
    MLIRCAPIIR
    MLIRPrepPipeDialect
    # MLIRPrepPipeTransforms
    MLIRPass
)