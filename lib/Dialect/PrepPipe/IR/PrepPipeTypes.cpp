#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::preppipe;
using namespace mlir::preppipe::PrepPipe;

#define GET_TYPEDEF_CLASSES
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeTypes.cpp.inc"
