// test_cedar_range.cpp - range_lookup: boundaries, gaps, and cross-segment scans
// range_lookup returns keys in [lo, hi] in sorted order (empty on an inverted range)

#include <gtest/gtest.h>

#include "li/cedar_index.hpp"

#include <vector>
#include <set>
#include <random>
#include <cstdint>

namespace {

using li::CedarIndex;
using li::Key;

CedarIndex mk(std::vector<Key> keys, double eps,
              std::size_t ws = 2048, std::size_t wm = 1024, int b = 11) {
    CedarIndex idx(eps, ws, wm, b);
    idx.build(keys);
    return idx;
}

std::vector<Key> expect_range(const std::vector<Key>& keys, Key lo, Key hi) {
    std::vector<Key> e;
    if (hi < lo) return e;
    for (Key k : keys) if (k >= lo && k <= hi) e.push_back(k);
    return e;
}

TEST(RangeLookup, FullyInside) {
    EXPECT_EQ(mk({10, 20, 30, 40, 50, 60}, 2.0).range_lookup(20, 50),
              (std::vector<Key>{20, 30, 40, 50}));
}

TEST(RangeLookup, CoversEverything) {
    EXPECT_EQ(mk({10, 20, 30, 40, 50}, 2.0).range_lookup(0, 1000),
              (std::vector<Key>{10, 20, 30, 40, 50}));
}

TEST(RangeLookup, HitsLastKey) {
    CedarIndex idx = mk({10, 20, 30, 40, 50}, 2.0);
    EXPECT_EQ(idx.range_lookup(40, 50), (std::vector<Key>{40, 50}));
    EXPECT_EQ(idx.range_lookup(40, 999), (std::vector<Key>{40, 50}));
}

TEST(RangeLookup, BoundsInGaps) {
    EXPECT_EQ(mk({10, 20, 30, 40, 50}, 2.0).range_lookup(15, 45),
              (std::vector<Key>{20, 30, 40}));
}

TEST(RangeLookup, EmptyGapRange) {
    EXPECT_TRUE(mk({10, 20, 30, 40, 50}, 2.0).range_lookup(21, 29).empty());
}

TEST(RangeLookup, EntirelyBelowAndAbove) {
    CedarIndex idx = mk({100, 200, 300}, 2.0);
    EXPECT_TRUE(idx.range_lookup(1, 50).empty());
    EXPECT_TRUE(idx.range_lookup(400, 999).empty());
}

TEST(RangeLookup, SingleKeyRange) {
    CedarIndex idx = mk({10, 20, 30, 40, 50}, 2.0);
    EXPECT_EQ(idx.range_lookup(30, 30), (std::vector<Key>{30}));
    EXPECT_TRUE(idx.range_lookup(25, 25).empty());
}

TEST(RangeLookup, InvertedRange) {
    EXPECT_TRUE(mk({10, 20, 30, 40, 50}, 2.0).range_lookup(40, 20).empty());
}

TEST(RangeLookup, EmptyIndex) {
    CedarIndex idx(2.0);
    idx.build({});
    EXPECT_TRUE(idx.range_lookup(0, 100).empty());
}

TEST(RangeLookup, SpansSegments) {
    std::vector<Key> keys;
    uint64_t v = 0;
    for (uint64_t i = 0; i < 300; ++i) { v += 1 + (i % 7); keys.push_back(v); }
    CedarIndex idx = mk(keys, 0.5);
    const Key lo = keys[100], hi = keys[200];
    EXPECT_EQ(idx.range_lookup(lo, hi), expect_range(keys, lo, hi));
}

// random cross-segment ranges after churn, including a small cap that fragments the key space into
// many segments (so a scan genuinely crosses segment boundaries)
TEST(RangeLookup, RandomAfterChurn) {
    for (double eps : {0.0, 0.5, 2.0}) {
        for (std::size_t ws : {std::size_t(8), std::size_t(2048)}) {
            std::mt19937_64 rng(1234 + uint64_t(eps * 10) + ws);
            std::set<Key> live;
            CedarIndex idx(eps, ws, ws / 2, ws >= 2048 ? 11 : 3);

            std::vector<Key> base;
            uint64_t v = 0;
            for (int i = 0; i < 60; ++i) { v += 1 + (rng() % 20); base.push_back(v); }
            idx.build(base);
            for (Key k : base) live.insert(k);

            std::uniform_int_distribution<uint64_t> kd(1, v + 300);
            for (int n = 0; n < 300; ++n) {
                if ((rng() & 1) && !live.empty()) {
                    auto it = live.begin();
                    std::advance(it, rng() % live.size());
                    idx.erase(*it);
                    live.erase(*it);
                } else {
                    Key k = kd(rng);
                    while (live.count(k)) k = kd(rng);
                    idx.insert(k, n);
                    live.insert(k);
                }
                if ((n % 20) == 0) {
                    std::vector<Key> sorted(live.begin(), live.end());
                    Key lo = kd(rng), hi = kd(rng);
                    if (hi < lo) std::swap(lo, hi);
                    ASSERT_EQ(idx.range_lookup(lo, hi), expect_range(sorted, lo, hi))
                        << "eps " << eps << " ws " << ws << " op " << n
                        << " range [" << lo << "," << hi << "]";
                }
            }
        }
    }
}

}