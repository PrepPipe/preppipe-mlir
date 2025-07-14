#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeDialect.h"
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeOps.h"
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::preppipe;
using namespace mlir::preppipe::PrepPipe;

#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeDialect.cpp.inc"

void PrepPipeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeOps.cpp.inc"

      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeTypes.cpp.inc"

      >();
}
