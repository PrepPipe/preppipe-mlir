// SPDX-FileCopyrightText: 2025 PrepPipe's Contributors
// SPDX-License-Identifier: Apache-2.0

#include "preppipe-mlir/Dialect/InputModel/IR/InputModel.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::preppipe::InputModel;

#include "preppipe-mlir/Dialect/InputModel/IR/InputModelBase.cpp.inc"

//===----------------------------------------------------------------------===
// InputModelDialect
//===----------------------------------------------------------------------===

void InputModelDialect::initialize() {
  // Register operations
  addOperations<
#define GET_OP_LIST
#include "preppipe-mlir/Dialect/InputModel/IR/InputModelOps.cpp.inc"
      >();
  
  // Register types
  addTypes<
#define GET_TYPEDEF_LIST
#include "preppipe-mlir/Dialect/InputModel/IR/InputModelTypes.cpp.inc"
      >();
  
  // Register attributes
  addAttributes<
#define GET_ATTRDEF_LIST
#include "preppipe-mlir/Dialect/InputModel/IR/InputModelAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===
// TableGen'd op method definitions
//===----------------------------------------------------------------------===

#define GET_OP_CLASSES
#include "preppipe-mlir/Dialect/InputModel/IR/InputModelOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "preppipe-mlir/Dialect/InputModel/IR/InputModelTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "preppipe-mlir/Dialect/InputModel/IR/InputModelAttrs.cpp.inc"
