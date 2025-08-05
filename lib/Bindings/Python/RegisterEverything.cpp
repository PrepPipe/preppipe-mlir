#include "preppipe-mlir-c/RegisterEverything.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/Pass/PassManager.h"

#include "preppipe-mlir-c/Dialect/PrepPipe.h"

PYBIND11_MODULE(_prepPipeMlirRegisterEverything, m) {
  m.doc() = "MLIR PrepPipe All Dialects, etc Registration";

  m.def("register_dialects", [](MlirContext context) {
    prepPipeMlirRegisterAllDialects(context);
  });

  m.def("register_translations", [](MlirContext context) {
    prepPipeMlirRegisterAllTranslations(context);
  });

  // Register all passes on load.
  prepPipeMlirRegisterAllPasses();
}
