
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "preppipe-mlir/InitAll.h"

using namespace mlir;

int main(int argc, char **argv) {
  // mlir::preppipe::registerAllPasses();

  DialectRegistry registry;
  mlir::preppipe::registerAllDialects(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "MLIR modular optimizer driver\n", registry));
}
