#include "preppipe-mlir/InitAll.h"
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipe.h"
#include "preppipe-mlir/Dialect/InputModel/IR/InputModel.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

void mlir::preppipe::registerAllDialects(mlir::DialectRegistry &registry) {
  // Register all dialects that this project produces and any dependencies.
  // Note: Arith dialect is registered separately where it's needed (e.g., in JSON frontend)
  // to avoid linker issues with template instantiation.
  registry.insert<mlir::preppipe::PrepPipe::PrepPipeDialect,
                  mlir::preppipe::InputModel::InputModelDialect
                  // mlir::preppipe::VNModelDialect
                  >();
}

void mlir::preppipe::registerAllPasses() {
  // Register all passes that this project produces.
  // This function can be expanded to include specific passes as needed.
}
