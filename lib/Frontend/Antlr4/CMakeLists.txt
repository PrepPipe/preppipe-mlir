set(GRAMMAR_TARGET CommandScan CommandParse SayScan)
set(ANTLR4_GENERATED_SRCS "")
set(ANTLR4_INCLUDE_DIRS "")

foreach(target IN LISTS GRAMMAR_TARGET)
  antlr4_generate(${target} "${CMAKE_CURRENT_SOURCE_DIR}/${target}.g4" BOTH)
  list(APPEND ANTLR4_GENERATED_SRCS "ANTLR4_SRC_FILES_${target}")
  list(APPEND ANTLR4_INCLUDE_DIRS "ANTLR4_INCLUDE_DIR_${target}")
endforeach()
