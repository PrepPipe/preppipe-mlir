#ifndef PREPPIPE_MLIR_INITALL_H
#define PREPPIPE_MLIR_INITALL_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace preppipe {

// Registers all dialects that this project needs to use.
void registerAllDialects(mlir::DialectRegistry &registry);

void registerAllPasses();

void registerAllTranslations();

} // namespace preppipe
} // namespace mlir

#endif // PREPPIPE_MLIR_INITALL_H
