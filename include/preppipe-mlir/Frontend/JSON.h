// SPDX-FileCopyrightText: 2025 PrepPipe's Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef PREPPIPE_FRONTEND_JSON_H
#define PREPPIPE_FRONTEND_JSON_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir {
namespace preppipe {
namespace frontend {

/// Read JSON files and convert them to InputModel Dialect MLIR.
/// \param context The MLIR context to use.
/// \param module The module to add the InputModel operations to.
/// \param filenames The list of JSON filenames to read.
/// \return true if successful, false otherwise.
bool readJSONFiles(MLIRContext &context, ModuleOp module, 
                   const std::vector<std::string> &filenames);

} // namespace frontend
} // namespace preppipe
} // namespace mlir

#endif // PREPPIPE_FRONTEND_JSON_H
