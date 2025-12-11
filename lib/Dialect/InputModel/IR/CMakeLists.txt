# SPDX-FileCopyrightText: 2025 PrepPipe's Contributors
# SPDX-License-Identifier: Apache-2.0

add_mlir_library(PrepPipeMLIRInputModelDialect
  InputModel.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/preppipe-mlir/Dialect/InputModel

  DEPENDS
  MLIRInputModelIncGen
  MLIRInputModelOpsIncGen
  MLIRInputModelAttrsIncGen
  MLIRInputModelTypesIncGen

  LINK_LIBS PUBLIC
  MLIRBytecodeOpInterface
  MLIRIR
  MLIRSupport
)

preppipe_mlir_target_includes(PrepPipeMLIRInputModelDialect)
