#include "li/status.hpp"
#include <gtest/gtest.h>
#include <string>

namespace {

TEST(Result, CarriesValueOrStatus) {
  li::Result<int> v{10};
  ASSERT_TRUE(v.ok());
  EXPECT_EQ(v.value(), 10);

  li::Result<int> e{li::Status::not_found};
  ASSERT_FALSE(e.ok());
  EXPECT_EQ(e.status(), li::Status::not_found);
}

TEST(Result, MovePreservesNonTrivialPayload) {
  li::Result<std::string> a{std::string("payload")};
  li::Result<std::string> b{std::move(a)};
  ASSERT_TRUE(b.ok());
  EXPECT_EQ(b.value(), "payload");
}

}