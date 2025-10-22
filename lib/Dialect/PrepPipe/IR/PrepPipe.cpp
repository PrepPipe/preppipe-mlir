#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipe.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::preppipe;
using namespace mlir::preppipe::PrepPipe;

#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeBase.cpp.inc"

#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeAttrs.cpp.inc"

#define GET_OP_CLASSES
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeOps.cpp.inc"

void PrepPipeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeOps.cpp.inc"

      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeTypes.cpp.inc"

      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeAttrs.cpp.inc"
      >();
}
