cmake_minimum_required(VERSION 3.20)

project(learned_index
    VERSION 0.0.1
    DESCRIPTION "A worst-case-bounded, updatable learned index"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(LI_BUILD_TESTS      "Build the invariant harness + unit tests" ON)
option(LI_BUILD_BENCH      "Build the microbenchmarks"                 OFF)
option(LI_INVARIANT_CHECKS "Compile the invariant asserts into code"  OFF)
option(LI_WERROR           "Treat warnings as errors (CI)"            OFF)

add_library(li INTERFACE)
add_library(li::li ALIAS li)
target_include_directories(li INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(li INTERFACE cxx_std_20)
if(LI_INVARIANT_CHECKS)
  target_compile_definitions(li INTERFACE LI_INVARIANT_CHECKS)
endif()

add_library(li_warnings INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(li_warnings INTERFACE -Wall -Wextra -Wpedantic)
  if(LI_WERROR)
    target_compile_options(li_warnings INTERFACE -Werror)
  endif()
endif()

if(LI_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()