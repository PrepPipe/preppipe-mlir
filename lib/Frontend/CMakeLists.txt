# SPDX-FileCopyrightText: 2025 PrepPipe's Contributors
# SPDX-License-Identifier: Apache-2.0

add_mlir_library(PrepPipeFrontendJSON
  JSON.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/preppipe-mlir/Frontend

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRArithDialect
  PrepPipeMLIRInputModelDialect
)

preppipe_mlir_target_includes(PrepPipeFrontendJSON)
