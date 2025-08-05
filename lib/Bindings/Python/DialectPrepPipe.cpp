#include "preppipe-mlir-c/Dialect/PrepPipe.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;
using namespace llvm;
using namespace mlir;
using namespace mlir::python::adaptors;

static void populateDialectPrepPipeSubmodule(const py::module &m) {
  // TODO
}

PYBIND11_MODULE(_prepPipeMlirDialectsPrepPipe, m) {
  m.doc() = "MLIR PrepPipe Dialect Python Bindings";
  populateDialectPrepPipeSubmodule(m);
}
