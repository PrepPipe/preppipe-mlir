#include "preppipe-mlir-c/RegisterEverything.h"
#include "preppipe-mlir/InitAll.h"
#include "mlir/CAPI/IR.h"

void prepPipeMlirRegisterAllDialects(MlirContext context)
{
  mlir::DialectRegistry registry;
  mlir::preppipe::registerAllDialects(registry);
  unwrap(context)->appendDialectRegistry(registry);
  unwrap(context)->loadAllAvailableDialects();
}

void prepPipeMlirRegisterAllPasses(void)
{
  mlir::preppipe::registerAllPasses();
}
void prepPipeMlirRegisterAllTranslations(MlirContext context)
{
  (void)context; // not used yet
  mlir::preppipe::registerAllTranslations();
}
