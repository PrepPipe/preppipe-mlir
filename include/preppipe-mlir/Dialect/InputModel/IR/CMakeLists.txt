# SPDX-FileCopyrightText: 2025 PrepPipe's Contributors
# SPDX-License-Identifier: Apache-2.0

set(LLVM_TARGET_DEFINITIONS InputModelBase.td)
mlir_tablegen(InputModelBase.h.inc -gen-dialect-decls -dialect=inputmodel)
mlir_tablegen(InputModelBase.cpp.inc -gen-dialect-defs -dialect=inputmodel)
add_public_tablegen_target(MLIRInputModelIncGen)

set(LLVM_TARGET_DEFINITIONS InputModelOps.td)
mlir_tablegen(InputModelOps.h.inc -gen-op-decls)
mlir_tablegen(InputModelOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRInputModelOpsIncGen)
add_dependencies(mlir-headers MLIRInputModelOpsIncGen)

set(LLVM_TARGET_DEFINITIONS InputModelAttrs.td)
mlir_tablegen(InputModelAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=inputmodel)
mlir_tablegen(InputModelAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=inputmodel)
add_public_tablegen_target(MLIRInputModelAttrsIncGen)
add_dependencies(mlir-headers MLIRInputModelAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS InputModelTypes.td)
mlir_tablegen(InputModelTypes.h.inc -gen-typedef-decls -typedefs-dialect=inputmodel)
mlir_tablegen(InputModelTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=inputmodel)
add_public_tablegen_target(MLIRInputModelTypesIncGen)
add_dependencies(mlir-headers MLIRInputModelTypesIncGen)
