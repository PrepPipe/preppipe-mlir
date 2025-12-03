# SPDX-FileCopyrightText: 2025 PrepPipe's Contributors
# SPDX-License-Identifier: Apache-2.0

# Add the AssetManager header to the include directory
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/../PrepPipe/IR/PrepPipeBase.td)

add_mlir_doc(PrepPipeOps PrepPipeOps -gen-op-doc "${CMAKE_CURRENT_BINARY_DIR}/PrepPipe/IR")

# No additional targets needed for header-only directory