cmake_minimum_required(VERSION 3.20)

project(cedar_index
    VERSION 0.0.1
    DESCRIPTION "A worst-case-bounded, updatable learned index"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CEDAR_BUILD_TESTS      "Build the invariant harness + unit tests" ON)
option(CEDAR_BUILD_BENCH      "Build the microbenchmarks"                OFF)
option(CEDAR_INVARIANT_CHECKS "Compile the invariant asserts into code"  OFF)
option(CEDAR_WERROR           "Treat warnings as errors (CI)"            OFF)

add_library(li INTERFACE)
add_library(li::li ALIAS li)
target_include_directories(li INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(li INTERFACE cxx_std_20)
if(CEDAR_INVARIANT_CHECKS)
  target_compile_definitions(li INTERFACE CEDAR_INVARIANT_CHECKS)
endif()

add_library(li_warnings INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(li_warnings INTERFACE -Wall -Wextra -Wpedantic)
  if(CEDAR_WERROR)
    target_compile_options(li_warnings INTERFACE -Werror)
  endif()
endif()

if(CEDAR_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()