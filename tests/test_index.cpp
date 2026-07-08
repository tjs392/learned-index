#include <gtest/gtest.h>
#include "li/index.hpp"
#include "li/status.hpp"

#include <vector>
#include <cstdint>

namespace {

using li::LearnedIndex;
using li::Status;

LearnedIndex make_index(std::vector<li::Key> keys, double eps) {
    LearnedIndex idx(eps);
    idx.build(std::move(keys));
    return idx;
}

void expect_all_keys_found(const std::vector<li::Key>& keys, double eps) {
    LearnedIndex idx = make_index(keys, eps);
    for (uint64_t i = 0; i < keys.size(); ++i) {
        auto r = idx.point_lookup(keys[i]);
        ASSERT_TRUE(r.ok())
            << "key " << keys[i] << " (true index " << i << ") not found at eps=" << eps;
        EXPECT_EQ(r.value(), i)
            << "key " << keys[i] << " looked up to wrong index at eps=" << eps;
    }
}

TEST(LearnedIndexDescriptor, ReturnsOwningSegmentForEveryKey) {
    std::vector<li::Key> keys;
    uint64_t v = 0;
    for (uint64_t i = 0; i < 300; ++i) { v += 1 + (i % 7); keys.push_back(v); }
    const double eps = 0.5;
    LearnedIndex idx = make_index(keys, eps);

    const auto& mt = idx.mapping_table_for_test();
    std::vector<uint64_t> base(mt.size(), 0);
    uint64_t acc = 0;
    for (size_t s = 0; s < mt.size(); ++s) { base[s] = acc; acc += mt[s].count; }

    for (uint64_t i = 0; i < keys.size(); ++i) {
        size_t di = idx.find_descriptor(keys[i]);
        const auto& d = mt[di];
        EXPECT_LE(d.key_low, keys[i]);
        EXPECT_GE(i, base[di]);
        EXPECT_LT(i, base[di] + d.count);
    }
}

TEST(LearnedIndexDescriptor, KeyBelowAllResolvesToFirst) {
    std::vector<li::Key> keys = {100, 200, 300, 400};
    LearnedIndex idx = make_index(keys, 1.0);
    EXPECT_EQ(idx.find_descriptor(50), 0u);
}

TEST(LearnedIndexLookup, ExactLine) {
    std::vector<li::Key> keys;
    for (uint64_t i = 0; i < 200; ++i) keys.push_back(1000 + i);
    expect_all_keys_found(keys, 4.0);
}

TEST(LearnedIndexLookup, CurvedMultiSegment) {
    std::vector<li::Key> keys;
    uint64_t v = 0;
    for (uint64_t i = 0; i < 500; ++i) { v += 1 + (i / 50); keys.push_back(v); }
    expect_all_keys_found(keys, 8.0);
}

TEST(LearnedIndexLookup, TightEpsManySegments) {
    std::vector<li::Key> keys;
    uint64_t v = 0;
    for (uint64_t i = 0; i < 300; ++i) { v += 1 + (i % 7); keys.push_back(v); }
    expect_all_keys_found(keys, 0.5);
}

TEST(LearnedIndexLookup, EpsZeroExact) {
    std::vector<li::Key> keys;
    for (uint64_t i = 0; i < 100; ++i) keys.push_back(5 + 3 * i);
    expect_all_keys_found(keys, 0.0);
}

TEST(LearnedIndexLookup, WideGaps) {
    std::vector<li::Key> keys;
    uint64_t v = 0;
    for (uint64_t i = 0; i < 200; ++i) { v += 1 + ((i * 40503u) % 500); keys.push_back(v); }
    expect_all_keys_found(keys, 2.0);
}

TEST(LearnedIndexLookup, NotFoundGapKey) {
    std::vector<li::Key> keys = {10, 20, 30, 40, 50};
    LearnedIndex idx = make_index(keys, 2.0);
    auto r = idx.point_lookup(25);
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(r.status(), Status::not_found);
}

TEST(LearnedIndexLookup, NotFoundBelowAll) {
    std::vector<li::Key> keys = {100, 200, 300};
    LearnedIndex idx = make_index(keys, 2.0);
    auto r = idx.point_lookup(1);
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(r.status(), Status::not_found);
}

TEST(LearnedIndexLookup, NotFoundAboveAll) {
    std::vector<li::Key> keys = {100, 200, 300};
    LearnedIndex idx = make_index(keys, 2.0);
    auto r = idx.point_lookup(9999);
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(r.status(), Status::not_found);
}

TEST(LearnedIndexLookup, FirstAndLastKey) {
    std::vector<li::Key> keys;
    uint64_t v = 0;
    for (uint64_t i = 0; i < 250; ++i) { v += 1 + (i % 5); keys.push_back(v); }
    LearnedIndex idx = make_index(keys, 1.0);

    auto rf = idx.point_lookup(keys.front());
    ASSERT_TRUE(rf.ok());
    EXPECT_EQ(rf.value(), 0u);

    auto rl = idx.point_lookup(keys.back());
    ASSERT_TRUE(rl.ok());
    EXPECT_EQ(rl.value(), keys.size() - 1);
}

TEST(LearnedIndexLookup, SingleKey) {
    std::vector<li::Key> keys = {42};
    LearnedIndex idx = make_index(keys, 4.0);
    auto r = idx.point_lookup(42);
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.value(), 0u);
    EXPECT_FALSE(idx.point_lookup(41).ok());
    EXPECT_FALSE(idx.point_lookup(43).ok());
}

TEST(LearnedIndexLookup, TwoKeysOneSegment) {
    std::vector<li::Key> keys = {42, 99};
    LearnedIndex idx = make_index(keys, 4.0);
    auto r0 = idx.point_lookup(42);
    auto r1 = idx.point_lookup(99);
    ASSERT_TRUE(r0.ok()); EXPECT_EQ(r0.value(), 0u);
    ASSERT_TRUE(r1.ok()); EXPECT_EQ(r1.value(), 1u);
}

TEST(LearnedIndexLookup, EmptyLearnedIndex) {
    LearnedIndex idx(4.0);
    idx.build({});
    auto r = idx.point_lookup(5);
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(r.status(), Status::not_found);
}

}