#include <gtest/gtest.h>
#include "li/hull_tree.hpp"
#include "li/model.hpp"

#include <vector>
#include <set>
#include <random>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <cstdint>

namespace {

using li::detail::HullTree;
using li::LinearModel;
using li::LeastSquaresSums;
using li::least_squares;
using li::Key;

LeastSquaresSums truth_moments(const std::vector<Key>& sorted_keys, Key origin) {
    LeastSquaresSums sums;
    for (std::size_t i = 0; i < sorted_keys.size(); ++i) {
        const double x = static_cast<double>(sorted_keys[i]) - static_cast<double>(origin);
        const double y = static_cast<double>(i);
        sums.add(x, y);
    }
    return sums;
}

void expect_moments_close(const LeastSquaresSums& got, const LeastSquaresSums& want) {
    ASSERT_EQ(got.n, want.n);
    const double rel = 1e-9;
    auto close = [&](double a, double b) {
        const double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
        return std::fabs(a - b) <= rel * scale;
    };
    EXPECT_TRUE(close(got.sum_x, want.sum_x))   << "sum_x "  << got.sum_x  << " vs " << want.sum_x;
    EXPECT_TRUE(close(got.sum_y, want.sum_y))   << "sum_y "  << got.sum_y  << " vs " << want.sum_y;
    EXPECT_TRUE(close(got.sum_xx, want.sum_xx)) << "sum_xx " << got.sum_xx << " vs " << want.sum_xx;
    EXPECT_TRUE(close(got.sum_xy, want.sum_xy)) << "sum_xy " << got.sum_xy << " vs " << want.sum_xy;
}

double line_at(const LinearModel& line, Key key, Key key_low) {
    return line.alpha * (static_cast<double>(key) - static_cast<double>(key_low)) + line.beta;
}

void expect_model_matches(const HullTree& tree, const std::vector<Key>& sorted_keys) {
    if (sorted_keys.empty()) {
        return;
    }
    const Key key_low = sorted_keys.front();

    const LeastSquaresSums want = truth_moments(sorted_keys, key_low);
    const LinearModel want_line = least_squares(want);
    const LinearModel got_line = tree.model(key_low);

    for (Key k : sorted_keys) {
        const double a = line_at(got_line, k, key_low);
        const double b = line_at(want_line, k, key_low);
        EXPECT_LE(std::fabs(a - b), 1e-3)
            << "model prediction mismatch at key " << k << " got " << a << " want " << b;
    }
}

void check_all(const HullTree& tree, const std::set<Key>& ref) {
    tree.validate();
    std::vector<Key> sorted(ref.begin(), ref.end());
    if (sorted.empty()) {
        return;
    }
    const Key origin = tree.moment_origin_for_test();
    expect_moments_close(tree.root_moments_for_test(), truth_moments(sorted, origin));
    expect_model_matches(tree, sorted);
}

TEST(HullMoments, ExactLineSingleSegment) {
    std::vector<Key> keys;
    for (Key k = 0; k < 300; ++k) {
        keys.push_back(1000 + 3 * k);
    }
    HullTree tree = HullTree::bulk_build(keys);
    std::set<Key> ref(keys.begin(), keys.end());
    check_all(tree, ref);

    const LinearModel line = tree.model(keys.front());
    EXPECT_NEAR(line.alpha, 1.0 / 3.0, 1e-9);
    EXPECT_NEAR(line.beta, 0.0, 1e-6);
}

TEST(HullMoments, SingleKey) {
    HullTree tree;
    tree.insert(42);
    const LinearModel line = tree.model(42);
    EXPECT_NEAR(line.alpha, 0.0, 1e-12);
    EXPECT_NEAR(line.beta, 0.0, 1e-12);
}

TEST(HullMoments, IncrementalMatchesBulk) {
    std::vector<Key> keys;
    for (Key k = 0; k < 400; ++k) {
        keys.push_back(k * 7 + 5);
    }
    HullTree bulk = HullTree::bulk_build(keys);

    HullTree incremental;
    for (Key k : keys) {
        incremental.insert(k);
    }

    const LinearModel a = bulk.model(keys.front());
    const LinearModel b = incremental.model(keys.front());
    EXPECT_NEAR(a.alpha, b.alpha, 1e-9);
    EXPECT_NEAR(a.beta, b.beta, 1e-6);
}

TEST(HullMoments, AscendingInsertsRebuildStress) {
    HullTree tree;
    std::set<Key> ref;
    for (Key k = 0; k < 600; ++k) {
        tree.insert(k * 2);
        ref.insert(k * 2);
    }
    check_all(tree, ref);
}

TEST(HullMoments, CurvedData) {
    std::vector<Key> keys;
    Key v = 0;
    for (Key i = 0; i < 500; ++i) {
        v += 1 + (i / 50);
        keys.push_back(v);
    }
    HullTree tree = HullTree::bulk_build(keys);
    std::set<Key> ref(keys.begin(), keys.end());
    check_all(tree, ref);
}

TEST(HullMoments, RandomChurn) {
    std::mt19937_64 rng(20260708);
    HullTree tree;
    std::set<Key> ref;
    std::uniform_int_distribution<Key> kd(0, 20000);

    for (int step = 0; step < 15000; ++step) {
        const bool ins = ref.empty() || (rng() & 1);
        if (ins) {
            Key k = kd(rng);
            while (ref.count(k)) {
                k = kd(rng);
            }
            tree.insert(k);
            ref.insert(k);
        } else {
            std::uniform_int_distribution<std::size_t> pick(0, ref.size() - 1);
            auto it = ref.begin();
            std::advance(it, static_cast<std::ptrdiff_t>(pick(rng)));
            tree.erase(*it);
            ref.erase(it);
        }
        if ((step % 200) == 0) {
            check_all(tree, ref);
        }
    }
    check_all(tree, ref);
}

TEST(HullMoments, ChurnBelowOriginAndMinDeletion) {
    HullTree tree;
    std::set<Key> ref;
    for (Key k = 0; k < 50; ++k) {
        tree.insert(1000 + k * 10);
        ref.insert(1000 + k * 10);
    }
    for (Key k = 0; k < 50; ++k) {
        tree.insert(k * 3);
        ref.insert(k * 3);
    }
    check_all(tree, ref);

    for (int i = 0; i < 40 && ref.size() > 1; ++i) {
        Key m = *ref.begin();
        tree.erase(m);
        ref.erase(m);
        check_all(tree, ref);
    }
}

}