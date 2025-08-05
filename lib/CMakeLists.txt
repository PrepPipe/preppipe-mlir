preppipe_mlir_enable_werror()

# add_subdirectory(Conversion)
add_subdirectory(Dialect)

set(LinkedLibs
  MLIRIR
  MLIRSupport

  # Dialects.
  MLIRPrepPipeDialect

  # Dialect passes.
  # TODO

  # Conversion passes.
  # TODO
)

add_mlir_library(PrepPipeMLIRInitAll
  InitAll.cpp

  LINK_LIBS PUBLIC
  ${LinkedLibs}
)

preppipe_mlir_target_includes(PrepPipeMLIRInitAll)
