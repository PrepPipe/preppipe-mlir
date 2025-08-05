#include "preppipe-mlir-c/Dialect/PrepPipe.h"

#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>

PYBIND11_MODULE(_prepPipeMlirPrepPipePasses, m) {
  m.doc() = "MLIR PrepPipe Passes Python Bindings";
  // we have no passes yet
  // mlirRegisterPrepPipePasses();
}
