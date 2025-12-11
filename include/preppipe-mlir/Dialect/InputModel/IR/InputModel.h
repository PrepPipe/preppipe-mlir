// SPDX-FileCopyrightText: 2025 PrepPipe's Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef PREPPIPE_INPUT_MODEL_H
#define PREPPIPE_INPUT_MODEL_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "preppipe-mlir/Dialect/InputModel/IR/InputModelBase.h.inc"

#define GET_TYPEDEF_CLASSES
#include "preppipe-mlir/Dialect/InputModel/IR/InputModelTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "preppipe-mlir/Dialect/InputModel/IR/InputModelAttrs.h.inc"

#define GET_OP_CLASSES
#include "preppipe-mlir/Dialect/InputModel/IR/InputModelOps.h.inc"

#endif // PREPPIPE_INPUT_MODEL_H
