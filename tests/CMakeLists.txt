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
  test_model.cpp
  test_segmentation.cpp
  test_index.cpp
  test_range.cpp
  test_pma.cpp
  test_inserts.cpp
  test_churn.cpp
  test_hull.cpp
  test_hull_tree.cpp
  test_hull_moments.cpp
)
target_link_libraries(li_tests PRIVATE
  li::li
  li_warnings
  GTest::gtest_main
)

target_compile_definitions(li_tests PRIVATE LI_INVARIANT_CHECKS)

include(GoogleTest)
gtest_discover_tests(li_tests)