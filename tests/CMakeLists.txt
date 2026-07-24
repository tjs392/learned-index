include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(li_tests
  test_status.cpp
  test_drift_guard.cpp
  test_segment_store.cpp
  test_segment_tree.cpp
  test_segmentation.cpp
  test_cedar_index.cpp
)

target_link_libraries(li_tests PRIVATE
  li::li
  li_warnings
  GTest::gtest_main
)

target_compile_definitions(li_tests PRIVATE CEDAR_INVARIANT_CHECKS)

include(GoogleTest)
gtest_discover_tests(li_tests)