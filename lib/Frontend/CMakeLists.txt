# SPDX-FileCopyrightText: 2025 PrepPipe's Contributors
# SPDX-License-Identifier: Apache-2.0

set(GRAMMAR_TARGET CommandScan CommandParse SayScan)
set(ANTLR4_GENERATED_SRCS "")
set(ANTLR4_INCLUDE_DIRS "")

foreach(target IN LISTS GRAMMAR_TARGET)
  antlr4_generate(${target} "${CMAKE_CURRENT_SOURCE_DIR}/Antlr4/${target}.g4" BOTH)
  set(ANTLR4_GENERATED_SRCS ${ANTLR4_GENERATED_SRCS} ${ANTLR4_SRC_FILES_${target}})
  set(ANTLR4_INCLUDE_DIRS ${ANTLR4_INCLUDE_DIRS} ${ANTLR4_INCLUDE_DIR_${target}})
endforeach()


add_mlir_library(PrepPipeFrontend
  JSON.cpp
  CommandScan.cpp
  ${ANTLR4_GENERATED_SRCS}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/preppipe-mlir/Frontend

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRArithDialect
  PrepPipeMLIRInputModelDialect
)
target_include_directories(PrepPipeFrontend PRIVATE ${ANTLR4_INCLUDE_DIRS})
target_link_libraries(PrepPipeFrontend PRIVATE antlr4_static)

preppipe_mlir_target_includes(PrepPipeFrontend)
