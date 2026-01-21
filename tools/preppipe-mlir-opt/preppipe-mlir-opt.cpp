
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "preppipe-mlir/InitAll.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

int main(int argc, char **argv) {
  // mlir::preppipe::registerAllPasses();

  DialectRegistry registry;
  mlir::preppipe::registerAllDialects(registry);

  // Enable location printing by default if not already specified
  llvm::SmallVector<const char *, 16> newArgv;
  newArgv.push_back(argv[0]); // program name

  bool hasDebugInfoFlag = false;
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    if (arg == "--mlir-print-debuginfo" || arg == "--mlir-pretty-debuginfo" ||
        arg.starts_with("--mlir-print-debuginfo=") ||
        arg.starts_with("--mlir-pretty-debuginfo=")) {
      hasDebugInfoFlag = true;
    }
    newArgv.push_back(argv[i]);
  }

  // Add --mlir-print-debuginfo if not already present
  if (!hasDebugInfoFlag) {
    newArgv.push_back("--mlir-print-debuginfo");
  }

  newArgv.push_back(nullptr); // null terminator

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      static_cast<int>(newArgv.size() - 1), const_cast<char **>(newArgv.data()),
      "MLIR modular optimizer driver\n", registry));
}
