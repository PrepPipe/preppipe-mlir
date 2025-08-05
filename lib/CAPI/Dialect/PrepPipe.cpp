#include "preppipe-mlir-c/Dialect/PrepPipe.h"
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipe.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace llvm;
using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(PREPPIPE, preppipe, preppipe::PrepPipe::PrepPipeDialect)
