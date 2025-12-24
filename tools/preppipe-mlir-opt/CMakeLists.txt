add_llvm_executable(preppipe-mlir-opt preppipe-mlir-opt.cpp)

install(TARGETS preppipe-mlir-opt
EXPORT PrepPipeMLIRTargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT preppipe-mlir-opt)

target_link_libraries(preppipe-mlir-opt PRIVATE
  MLIROptLib
  MLIRTransforms
  MLIRArithDialect
  PrepPipeMLIRInitAll
)
