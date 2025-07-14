# CMakeLists.txt adapted from torch-mlir

#-------------------------------------------------------------------------------
# Project setup and globals
#-------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.12)

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

project(preppipe-mlir LANGUAGES CXX C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

include(CMakeDependentOption)

#-------------------------------------------------------------------------------
# Project options
#-------------------------------------------------------------------------------

option(PREPPIPE_MLIR_ENABLE_WERROR_FLAG "Enable `-Werror` flag on supported directories, treat warning as error" OFF)

set(PREPPIPE_MLIR_TABLEGEN_FLAGS "")

option(PREPPIPE_MLIR_OUT_OF_TREE_BUILD "Specifies an out of tree build" OFF)

macro(preppipe_mlir_enable_werror)
  if(PREPPIPE_MLIR_ENABLE_WERROR_FLAG)
    if(NOT MSVC)
      add_compile_options(-Werror)
    endif()
  endif()
endmacro()

if(MSVC)
  add_definitions(-D_USE_MATH_DEFINES)
endif()

#-------------------------------------------------------------------------------
# Configure out-of-tree vs in-tree build
#-------------------------------------------------------------------------------

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR OR PREPPIPE_MLIR_OUT_OF_TREE_BUILD)
  message(STATUS "preppipe-mlir out-of-tree build.")
  # Out-of-tree build

  #-------------------------------------------------------------------------------
  # MLIR/LLVM Configuration
  #-------------------------------------------------------------------------------

  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

  # Define the default arguments to use with 'lit', and an option for the user to
  # override.
  set(LIT_ARGS_DEFAULT "-sv")
  if (MSVC_IDE OR XCODE)
    set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
  endif()
  set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  #TODO: support python bindings in out-of-tree builds
  set(MLIR_ENABLE_BINDINGS_PYTHON ON)

  set(PREPPIPE_MLIR_BUILT_STANDALONE ON)
  set(BACKEND_PACKAGE_STRING "LLVM ${LLVM_PACKAGE_VERSION}")
else()
  message(STATUS "preppipe-mlir in-tree build.")
  # In-tree build with LLVM_EXTERNAL_PROJECTS=preppipe-mlir

  option(MLIR_ENABLE_BINDINGS_PYTHON "Enables MLIR Python Bindings" OFF)

  # TODO: Fix this upstream so that global include directories are not needed.
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
endif()

set(PREPPIPE_MLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(PREPPIPE_MLIR_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "Building preppipe-mlir project at ${PREPPIPE_MLIR_SOURCE_DIR} (into ${PREPPIPE_MLIR_BINARY_DIR})")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

function(preppipe_mlir_target_includes target)
  set(_dirs
    $<BUILD_INTERFACE:${MLIR_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${MLIR_GENERATED_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${PREPPIPE_MLIR_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PREPPIPE_MLIR_BINARY_DIR}/include>
  )
  # In LLVM parlance, the actual target may just be an interface and may not
  # be responsible for actually compiling anything. The corresponding obj.
  # target, when present, is just used for compilation and does not
  # contribute to the interface properties.
  # TODO: Normalize this upstream.
  target_include_directories(${target} PUBLIC ${_dirs})
  if(TARGET obj.${target})
    target_include_directories(obj.${target} PRIVATE ${_dirs})
  endif()
endfunction()

# Configure CMake.
list(APPEND CMAKE_MODULE_PATH ${MLIR_MAIN_SRC_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${LLVM_MAIN_SRC_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/build_tools/cmake)

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddMLIRPython)

################################################################################
# Setup python.
################################################################################

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

#add_custom_target(check-preppipe-mlir-all)
#add_dependencies(check-preppipe-mlir-all check-preppipe-mlir)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  # If parent projects want to configure where to place the python packages,
  # respect that.
  if(NOT PREPPIPE_MLIR_PYTHON_PACKAGES_DIR)
    set(PREPPIPE_MLIR_PYTHON_PACKAGES_DIR "${CMAKE_CURRENT_BINARY_DIR}/python_packages")
  endif()
endif()

#add_subdirectory(test)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(DIRECTORY include/preppipe-mlir include/preppipe-mlir-c
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
          COMPONENT preppipe-mlir-headers
          FILES_MATCHING
          PATTERN "*.def"
          PATTERN "*.h"
          PATTERN "*.inc"
          PATTERN "*.td"
          PATTERN "LICENSE.TXT"
          )

  install(DIRECTORY ${PREPPIPE_MLIR_BINARY_DIR}/include/preppipe-mlir
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
          COMPONENT preppipe-mlir-headers
          FILES_MATCHING
          PATTERN "*.def"
          PATTERN "*.h"
          PATTERN "*.gen"
          PATTERN "*.inc"
          PATTERN "*.td"
          PATTERN "CMakeFiles" EXCLUDE
          PATTERN "config.h" EXCLUDE
          )

  if (NOT LLVM_ENABLE_IDE)
    add_llvm_install_targets(install-preppipe-mlir-headers
            DEPENDS preppipe-mlir-headers
            COMPONENT preppipe-mlir-headers)
  endif()
endif()

#-------------------------------------------------------------------------------
# Sub-projects
#-------------------------------------------------------------------------------

# Finish with top-level Python bindings so it can handle additional deps.
if(MLIR_ENABLE_BINDINGS_PYTHON)
#  add_subdirectory(python)
endif()
