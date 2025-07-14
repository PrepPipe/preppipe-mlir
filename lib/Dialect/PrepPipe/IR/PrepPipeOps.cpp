//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
//
// This file is meant to include the `TorchOps.cpp.inc` file and compile it
// separately from the main TorchOps.cpp file. The .inc file takes a very long
// time to compile, and slows down the iteration time on folders,
// canonicalizations, parser/printers, etc. in the actual TorchOps.cpp file, so
// it makes sense to isolate it and let the build system cache it.
//
//===----------------------------------------------------------------------===//

#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::preppipe;
using namespace mlir::preppipe::PrepPipe;

#define GET_OP_CLASSES
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeOps.cpp.inc"
