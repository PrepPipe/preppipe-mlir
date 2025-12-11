
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/ToolUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include "preppipe-mlir/InitAll.h"
#include "preppipe-mlir/Frontend/JSON.h"

using namespace mlir;

static llvm::cl::list<std::string> inputFilenames(
    llvm::cl::Positional, llvm::cl::desc("<input files>"), llvm::cl::OneOrMore);

static llvm::cl::opt<std::string> outputPath(
    "o", llvm::cl::desc("Output path (file or directory)"), llvm::cl::init(""));

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Parse command line arguments
  if (llvm::cl::ParseCommandLineOptions(argc, argv, "PrepPipe Pipeline Tool\n")) {
    return 1;
  }

  // Create MLIR context and registry
  MLIRContext context;
  DialectRegistry registry;
  preppipe::registerAllDialects(registry);
  context.appendDialectRegistry(registry);

  // Create a new module
  auto module = ModuleOp::create(UnknownLoc::get(&context));

  // Read JSON files and convert to InputModel MLIR
  if (!preppipe::frontend::readJSONFiles(context, module, inputFilenames)) {
    llvm::errs() << "Error reading JSON files\n";
    return 1;
  }

  // Verify the module
  if (failed(verify(module))) {
    llvm::errs() << "Error verifying module\n";
    return 1;
  }

  // Write the module to the output file or stdout
  if (outputPath.empty()) {
    // Write to stdout
    module.print(llvm::outs());
  } else {
    // Write to file
    std::error_code ec;
    llvm::raw_fd_ostream outputFile(outputPath, ec);
    if (ec) {
      llvm::errs() << "Error opening output file: " << ec.message() << "\n";
      return 1;
    }
    module.print(outputFile);
  }

  return 0;
}
