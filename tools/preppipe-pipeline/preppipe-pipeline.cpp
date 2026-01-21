
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/ToolUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include "preppipe-mlir/InitAll.h"
#include "preppipe-mlir/Frontend/JSON.h"
#include "preppipe-mlir/Dialect/InputModel/IR/InputModel.h"
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipe.h"

using namespace mlir;

static llvm::cl::list<std::string> inputFilenames(
    llvm::cl::Positional, llvm::cl::desc("<input files>"), llvm::cl::OneOrMore);

static llvm::cl::opt<std::string> outputPath(
    "o", llvm::cl::desc("Output path (file or directory)"), llvm::cl::init(""));

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Parse command line arguments
  // ParseCommandLineOptions returns false on error, true on success
  if (!llvm::cl::ParseCommandLineOptions(argc, argv, "PrepPipe Pipeline Tool\n")) {
    return 1;
  }

  // Check if input files were provided
  if (inputFilenames.empty()) {
    llvm::errs() << "Error: No input files specified\n";
    llvm::errs() << "Usage: " << argv[0] << " [options] <input files>\n";
    llvm::errs() << "See: " << argv[0] << " --help\n";
    return 1;
  }

  // Create MLIR context and registry
  MLIRContext context;
  DialectRegistry registry;
  preppipe::registerAllDialects(registry);
  context.appendDialectRegistry(registry);

  // Load the dialects we need
  context.loadDialect<preppipe::InputModel::InputModelDialect>();
  context.loadDialect<preppipe::PrepPipe::PrepPipeDialect>();

  // Create a new module
  auto module = ModuleOp::create(UnknownLoc::get(&context));

  // Read JSON files and convert to InputModel MLIR
  if (!preppipe::frontend::readJSONFiles(context, module, inputFilenames)) {
    llvm::errs() << "Error: Failed to read JSON files\n";
    return 1;
  }

  // Verify the module
  OpPrintingFlags flags;
  flags.enableDebugInfo(true, /*prettyForm=*/false);

  if (failed(verify(module))) {
    llvm::errs() << "Error: Module verification failed\n";
    module.print(llvm::errs(), flags);
    return 1;
  }

  // Write the module to the output file or stdout
  if (outputPath.empty()) {
    // Write to stdout
    module.print(llvm::outs(), flags);
    llvm::outs() << "\n";
  } else {
    // Write to file
    std::error_code ec;
    llvm::raw_fd_ostream outputFile(outputPath, ec);
    if (ec) {
      llvm::errs() << "Error: Failed to open output file '" << outputPath
                   << "': " << ec.message() << "\n";
      return 1;
    }
    module.print(outputFile, flags);
    outputFile << "\n";
  }

  return 0;
}
