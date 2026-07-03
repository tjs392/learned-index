#include "li/model.hpp"
#include <gtest/gtest.h>

TEST(Model, Constructs) {
  li::Model m{1.0, 0.0};
  EXPECT_EQ(m.alpha, 1.0);
  EXPECT_EQ(m.beta, 0.0);
}