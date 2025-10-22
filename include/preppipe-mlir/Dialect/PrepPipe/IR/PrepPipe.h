#ifndef PREPPIPE_PREPPIPE_H
#define PREPPIPE_PREPPIPE_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeBase.h.inc"
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeAttrs.h.inc"

#define GET_OP_CLASSES
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeOps.h.inc"

#endif // PREPPIPE_PREPPIPE_H
