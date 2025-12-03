# SPDX-FileCopyrightText: 2025 PrepPipe's Contributors
# SPDX-License-Identifier: Apache-2.0

# Add the AssetManager implementation
add_mlir_library(PreppipeAsset
  AssetManager.cpp

  DEPENDS
  MLIRPrepPipeOpsIncGen

  LINK_COMPONENTS
  Support
  
  LINK_LIBS PUBLIC
  MLIRIR
)

# 添加包含目录
target_include_directories(PreppipeAsset
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
