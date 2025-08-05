#ifndef PREPPIPE_MLIR_REGISTER_EVERYTHING_H
#define PREPPIPE_MLIR_REGISTER_EVERYTHING_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void prepPipeMlirRegisterAllDialects(MlirContext context);
MLIR_CAPI_EXPORTED void prepPipeMlirRegisterAllPasses(void);
MLIR_CAPI_EXPORTED void prepPipeMlirRegisterAllTranslations(MlirContext context);

#ifdef __cplusplus
}
#endif
#endif // PREPPIPE_MLIR_REGISTER_EVERYTHING_H
