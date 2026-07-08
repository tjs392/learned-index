#include "li/model.hpp"
#include <gtest/gtest.h>

namespace {

using li::least_squares;
using li::LinearModel;
using li::LeastSquaresSums;
using li::predict;
using li::residual;

LeastSquaresSums moments_from(const std::vector<li::Key> &keys, li::Key key_low) {
    LeastSquaresSums m{0, 0.0, 0.0, 0.0, 0.0};
    for (uint64_t rank = 0; rank < keys.size(); ++rank) {
        double x = static_cast<double>(keys[rank] - key_low);
        double y = static_cast<double>(rank);
        m.n += 1;
        m.sum_x += x;
        m.sum_y += y;
        m.sum_xx += x * x;
        m.sum_xy += x * y;
    }
    return m;
}

TEST(LinearModel, RecoversKnownLine) {
    const li::Key key_low = 1000;
    std::vector<li::Key> keys;
    for (uint64_t i = 0; i < 8; ++i) keys.push_back(key_low + 2 * i);

    LinearModel m = least_squares(moments_from(keys, key_low));

    EXPECT_NEAR(m.alpha, 0.5, 1e-9);
    EXPECT_NEAR(m.beta,  0.0, 1e-9);
}

TEST(LinearModel, PredictInLocalFrame) {
    const li::Key key_low = 1000;
    std::vector<li::Key> keys;
    for (uint64_t i = 0; i < 8; ++i) keys.push_back(key_low + 2 * i);
    LinearModel m = least_squares(moments_from(keys, key_low));

    EXPECT_EQ(predict(m, key_low + 6, key_low), 3u);
    EXPECT_EQ(predict(m, key_low, key_low), 0u);
}

TEST(LinearModel, ResidualsZeroOnExactLine) {
    const li::Key key_low = 1000;
    std::vector<li::Key> keys;
    for (uint64_t i = 0; i < 8; ++i) keys.push_back(key_low + 2 * i);
    LinearModel m = least_squares(moments_from(keys, key_low));

    for (uint64_t i = 0; i < keys.size(); ++i) {
        EXPECT_NEAR(residual(m, keys[i], key_low, i), 0.0, 1e-9);
    }
}

TEST(LinearModel, SinglePointHorizontal) {
    const li::Key key_low = 500;
    LeastSquaresSums m1 = moments_from({key_low}, key_low);
    LinearModel m = least_squares(m1);

    EXPECT_EQ(m.alpha, 0.0);
    EXPECT_NEAR(m.beta, 0.0, 1e-9);
}

}
