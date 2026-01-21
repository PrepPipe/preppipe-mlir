# CMakeLists.txt adapted from torch-mlir

# There are two ways to get dependencies:
# 1. Use system dependencies: install dependencies and set the appropriate variables; see the CI scripts for details.
# 2. Build dependencies from source. You will need to:
#       (1) run "git submodule update --init --recursive" to get all dependencies (they are git submodules)
#       (2) run externals/fetch_files.py script to download files that are not git submodules (e.g., antlr4 jar)
#    You don't need to run these commands if you have all the dependencies installed. See README.md for build instructions.
#-------------------------------------------------------------------------------
# Project setup and globals
#-------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.15)

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
option(MLIR_ENABLE_BINDINGS_PYTHON "Enables MLIR Python Bindings" OFF)

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
# Helpers to handle external dependencies
#
# All these functions will use existing files, etc if already downloaded
# So it is ok to run this script at configure time.
#-------------------------------------------------------------------------------

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# filename should be within the externals directory
function(fetch_file filename)
  execute_process(
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/externals/fetch_files.py
      ${filename}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/externals
    RESULT_VARIABLE _fetch_result
    COMMAND_ECHO STDOUT
  )
  if (NOT _fetch_result EQUAL 0)
    message(FATAL_ERROR "Cannot download ${filename}. Please either retry or manually download.")
  endif()
endfunction()

# submodule_path should be within the externals directory
function(fetch_submodule submodule_path)
  execute_process(
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/externals/init_submodule.py
      "externals/${submodule_path}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE _fetch_result
    COMMAND_ECHO STDOUT
  )
  if (NOT _fetch_result EQUAL 0)
    message(FATAL_ERROR "Cannot fetch ${submodule_path}. Please either retry or manually download.")
  endif()
endfunction()

#-------------------------------------------------------------------------------
# Actual handling of external dependencies
#-------------------------------------------------------------------------------

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/externals/install)

# antlr4
# we need:
# 1. ANTLR4_JAR_LOCATION pointing to antlr4 complete jar
# 2. antlr4-runtime, either installed in the system or built from source (in externals folder)
if (USE_SYSTEM_ANTLR4)
  if(NOT DEFINED ANTLR4_JAR_LOCATION)
    message(FATAL_ERROR "Please specify the location of antlr4 jar")
  endif()
  find_package(antlr4-runtime CONFIG REQUIRED)
  find_package(antlr4-generator CONFIG REQUIRED)
else()
  fetch_file(antlr.jar)
  find_file(ANTLR4_JAR_LOCATION NAMES antlr.jar PATHS ${CMAKE_CURRENT_SOURCE_DIR}/externals)
  fetch_submodule(antlr4-runtime-wrapper/antlr4)
  find_package(antlr4-runtime CONFIG)
  # We need to build antlr4-runtime at configuration time to use antlr4-generator cmake functions.
  if(NOT antlr4-runtime_FOUND)
    message(STATUS "Building antlr4-runtime from source")
    execute_process(
      COMMAND
        ${CMAKE_COMMAND}
        -G ${CMAKE_GENERATOR}
        "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
        "-DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/externals/install"
        -S ${CMAKE_CURRENT_SOURCE_DIR}/externals/antlr4-runtime-wrapper/antlr4/runtime/Cpp
        -B ${CMAKE_CURRENT_SOURCE_DIR}/externals/build/antlr4-runtime
        -DANTLR4_INSTALL=ON
        -DWITH_DEMO=OFF
      COMMAND
        ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_SOURCE_DIR}/externals/build/antlr4-runtime --target install
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/externals
      COMMAND_ECHO STDOUT
      COMMAND_ERROR_IS_FATAL ANY
    )
    find_package(antlr4-runtime CONFIG REQUIRED)
  endif()
  find_package(antlr4-generator CONFIG REQUIRED)
endif()
message(STATUS "Using ANTLR4=\"${ANTLR4}\"")

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

  set(PREPPIPE_MLIR_BUILT_STANDALONE ON)
  set(BACKEND_PACKAGE_STRING "LLVM ${LLVM_PACKAGE_VERSION}")
else()
  message(STATUS "preppipe-mlir in-tree build.")
  # In-tree build with LLVM_EXTERNAL_PROJECTS=preppipe-mlir

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

add_subdirectory(test)

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
