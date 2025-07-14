#ifndef PREPPIPE_PREPPIPE_OPS_H
#define PREPPIPE_PREPPIPE_OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#define GET_OP_CLASSES
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeOps.h.inc"

#endif // PREPPIPE_PREPPIPE_OPS_H
