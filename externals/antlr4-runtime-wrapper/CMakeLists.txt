# At the time of writing (antlr4 4.13.2), antlr4-cpp-runtime's root
# CMakeLists.txt does not use project() command, but the one in the
# subdirectory uses ${PROJECT_SOURCE_DIR}, so if we try to add_subdirectory()
# to antlr4-cpp-runtime, it will fail.
# We use a wrapper here to add the project() command.

cmake_minimum_required (VERSION 3.15)
project(antlr4-runtime-wrapper LANGUAGES CXX)
add_subdirectory(antlr4/runtime/Cpp)
