#ifndef PREPPIPE_MLIR_C_DIALECT_PREPPIPE_H
#define PREPPIPE_MLIR_C_DIALECT_PREPPIPE_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(PREPPIPE, preppipe);

#ifdef __cplusplus
}
#endif

// uncomment the following lines after we introduce the C API for passes
// #include "preppipe-mlir/Dialect/PrepPipe/Transforms/Passes/capi.h.inc"

#endif // PREPPIPE_MLIR_C_DIALECT_PREPPIPE_H
