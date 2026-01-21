add_llvm_executable(preppipe-pipeline preppipe-pipeline.cpp)

install(TARGETS preppipe-pipeline
EXPORT PrepPipeMLIRTargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT preppipe-pipeline)

target_link_libraries(preppipe-pipeline PRIVATE
  MLIROptLib
  MLIRTransforms
  MLIRArithDialect
  PrepPipeMLIRInitAll
  PrepPipeFrontend
)
