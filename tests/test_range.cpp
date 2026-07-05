#include <gtest/gtest.h>
#include "li/index.hpp"
#include <vector>
#include <cstdint>

namespace {

using li::Index;

Index make_index(std::vector<li::Key> keys, double eps) {
    Index idx(eps);
    idx.build(std::move(keys));
    return idx;
}

std::vector<li::Key> collect(const li::RangeView& v) {
    std::vector<li::Key> out;
    for (li::Key k : v) out.push_back(k);
    return out;
}

TEST(RangeLookup, FullyInside) {
    Index idx = make_index({10, 20, 30, 40, 50, 60}, 2.0);
    EXPECT_EQ(collect(idx.range_lookup(20, 50)), (std::vector<li::Key>{20, 30, 40, 50}));
}

TEST(RangeLookup, CoversEverything) {
    Index idx = make_index({10, 20, 30, 40, 50}, 2.0);
    EXPECT_EQ(collect(idx.range_lookup(0, 1000)), (std::vector<li::Key>{10, 20, 30, 40, 50}));
}

TEST(RangeLookup, HitsLastKey) {
    Index idx = make_index({10, 20, 30, 40, 50}, 2.0);
    EXPECT_EQ(collect(idx.range_lookup(40, 50)), (std::vector<li::Key>{40, 50}));
    EXPECT_EQ(collect(idx.range_lookup(40, 999)), (std::vector<li::Key>{40, 50}));
}

TEST(RangeLookup, BoundsInGaps) {
    Index idx = make_index({10, 20, 30, 40, 50}, 2.0);
    EXPECT_EQ(collect(idx.range_lookup(15, 45)), (std::vector<li::Key>{20, 30, 40}));
}

TEST(RangeLookup, EmptyGapRange) {
    Index idx = make_index({10, 20, 30, 40, 50}, 2.0);
    EXPECT_TRUE(collect(idx.range_lookup(21, 29)).empty());
}

TEST(RangeLookup, EntirelyBelowAndAbove) {
    Index idx = make_index({100, 200, 300}, 2.0);
    EXPECT_TRUE(collect(idx.range_lookup(1, 50)).empty());
    EXPECT_TRUE(collect(idx.range_lookup(400, 999)).empty());
}

TEST(RangeLookup, SingleKeyRange) {
    Index idx = make_index({10, 20, 30, 40, 50}, 2.0);
    EXPECT_EQ(collect(idx.range_lookup(30, 30)), (std::vector<li::Key>{30}));
    EXPECT_TRUE(collect(idx.range_lookup(25, 25)).empty());
}

TEST(RangeLookup, InvertedRange) {
    Index idx = make_index({10, 20, 30, 40, 50}, 2.0);
    EXPECT_TRUE(collect(idx.range_lookup(40, 20)).empty());
}

TEST(RangeLookup, EmptyIndex) {
    Index idx(2.0);
    idx.build({});
    EXPECT_TRUE(collect(idx.range_lookup(0, 100)).empty());
}

TEST(RangeLookup, SpansSegments) {
    std::vector<li::Key> keys;
    uint64_t v = 0;
    for (uint64_t i = 0; i < 300; ++i) { v += 1 + (i % 7); keys.push_back(v); }
    Index idx = make_index(keys, 0.5);
    const li::Key lo = keys[100], hi = keys[200];
    std::vector<li::Key> expected;
    for (li::Key k : keys) if (k >= lo && k <= hi) expected.push_back(k);
    EXPECT_EQ(collect(idx.range_lookup(lo, hi)), expected);
}

}