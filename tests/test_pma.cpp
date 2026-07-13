#include <gtest/gtest.h>
#include "li/pma.hpp"
#include <vector>
#include <set>
#include <random>
#include <cstdint>

namespace {
using li::detail::PmaBlock;
using li::Key;
using li::Rank;
using Block = PmaBlock;

Block make_block(std::vector<Key> keys) {
    std::vector<Key> payloads;
    for (Key k : keys) payloads.push_back(k + 7);
    return Block::bulk_load(keys, payloads, PmaBlock::capacity_for_key_cap(1u << 20));
}

std::vector<Key> collect(const Block& b) {
    std::vector<Key> out;
    b.scan(0, ~Key(0), [&](Key k, Key) { out.push_back(k); });
    return out;
}

std::vector<Key> collect_range(const Block& b, Key lo, Key hi) {
    std::vector<Key> out;
    b.scan(lo, hi, [&](Key k, Key) { out.push_back(k); });
    return out;
}

TEST(PmaRankSlot, RoundTrip) {
    std::vector<Key> keys;
    for (Key i = 0; i < 64; ++i) keys.push_back(i * 10);
    Block b = make_block(keys);
    ASSERT_EQ(b.size(), 64u);
    for (Rank r = 0; r < 64; ++r) {
        std::size_t s = b.slot_of_rank(r);
        EXPECT_EQ(b.local_rank(s), r);
        EXPECT_EQ(b.key_at(s), r * 10);
        EXPECT_EQ(b.payload_at(s), r * 10 + 7);
    }
}

TEST(PmaRankSlot, EndSentinels) {
    Block b = make_block({1, 2, 3, 4, 5});
    EXPECT_EQ(b.local_rank(b.capacity()), b.size());
    EXPECT_EQ(b.slot_of_rank(b.size()), b.capacity());
    EXPECT_EQ(b.slot_of_rank(999), b.capacity());
}

TEST(PmaRankSlot, SingleKey) {
    Block b = make_block({42});
    EXPECT_EQ(b.size(), 1u);
    EXPECT_EQ(b.local_rank(b.slot_of_rank(0)), 0u);
    EXPECT_EQ(b.key_at(b.slot_of_rank(0)), 42u);
}

TEST(PmaFind, PresentAndAbsent) {
    Block b = make_block({10, 20, 30, 40, 50});
    for (Key k : {10, 20, 30, 40, 50}) EXPECT_TRUE(b.find(k).has_value());
    for (Key k : {0, 5, 25, 55, 1000}) EXPECT_FALSE(b.find(k).has_value());
}

TEST(PmaFind, LowerBoundSemantics) {
    Block b = make_block({10, 20, 30, 40, 50});
    EXPECT_EQ(b.key_at(b.lower_bound(25)), 30u);
    EXPECT_EQ(b.key_at(b.lower_bound(30)), 30u);
    EXPECT_EQ(b.key_at(b.lower_bound(10)), 10u);
    EXPECT_EQ(b.lower_bound(51), b.capacity());
}

TEST(PmaInsert, InteriorPreservesOrder) {
    Block b = make_block({10, 20, 30, 40, 50});
    b.insert(25, 32);
    b.insert(5, 12);
    b.insert(45, 52);
    b.check_invariants();
    EXPECT_EQ(collect(b), (std::vector<Key>{5, 10, 20, 25, 30, 40, 45, 50}));
    EXPECT_EQ(b.payload_at(*b.find(25)), 32u);
}

TEST(PmaInsert, TriggersGrow) {
    Block b = make_block({});
    std::size_t cap0 = b.capacity();
    for (Key v = 0; v < 2000; ++v) b.insert(v, v);
    b.check_invariants();
    EXPECT_GT(b.capacity(), cap0);
    for (Key v = 0; v < 2000; ++v) EXPECT_TRUE(b.find(v).has_value());
}

TEST(PmaInsert, IntoEmptyLeafRegression) {
    Block b = make_block({1000});
    b.insert(1, 1);
    b.insert(2, 2);
    b.insert(3, 3);
    b.check_invariants();
    EXPECT_EQ(b.size(), 4u);
    EXPECT_EQ(collect(b), (std::vector<Key>{1, 2, 3, 1000}));
}

TEST(PmaInsert, ManyRandomStayConsistent) {
    Block b = make_block({});
    std::mt19937_64 rng(7);
    std::set<Key> ref;
    for (int i = 0; i < 3000; ++i) {
        Key k = rng() % 100000;
        if (ref.insert(k).second) b.insert(k, k);
    }
    b.check_invariants();
    std::vector<Key> expected(ref.begin(), ref.end());
    EXPECT_EQ(collect(b), expected);
}

TEST(PmaAppend, FastPathIncreasing) {
    Block b = make_block({0, 1, 2});
    for (Key k = 3; k < 1000; ++k) b.append(k, k);
    b.check_invariants();
    EXPECT_EQ(b.size(), 1000u);
    for (Key k = 0; k < 1000; ++k) EXPECT_TRUE(b.find(k).has_value());
}

TEST(PmaAppend, MatchesInsertResult) {
    Block a = make_block({});
    Block c = make_block({});
    for (Key k = 1; k <= 500; ++k) { a.append(k, k); c.insert(k, k); }
    EXPECT_EQ(collect(a), collect(c));
}

TEST(PmaErase, RemovePresentAndAbsent) {
    Block b = make_block({10, 20, 30, 40, 50});
    EXPECT_TRUE(b.erase(30).found);
    EXPECT_FALSE(b.erase(30).found);
    EXPECT_FALSE(b.erase(999).found);
    EXPECT_EQ(collect(b), (std::vector<Key>{10, 20, 40, 50}));
}

TEST(PmaErase, ShrinksWhenSparse) {
    std::vector<Key> keys;
    for (Key i = 0; i < 400; ++i) keys.push_back(i * 2);
    Block b = make_block(keys);
    std::size_t cap0 = b.capacity();
    for (Key i = 0; i < 380; ++i) b.erase(i * 2);
    b.check_invariants();
    EXPECT_LT(b.capacity(), cap0);
    EXPECT_EQ(b.size(), 20u);
}

TEST(PmaScan, HalfOpenRange) {
    Block b = make_block({10, 20, 30, 40, 50});
    EXPECT_EQ(collect_range(b, 20, 50), (std::vector<Key>{20, 30, 40}));
    EXPECT_EQ(collect_range(b, 20, 51), (std::vector<Key>{20, 30, 40, 50}));
    EXPECT_EQ(collect_range(b, 15, 45), (std::vector<Key>{20, 30, 40}));
}

TEST(PmaScan, EmptyAndBoundaryRanges) {
    Block b = make_block({10, 20, 30, 40, 50});
    EXPECT_TRUE(collect_range(b, 21, 29).empty());
    EXPECT_TRUE(collect_range(b, 100, 200).empty());
    EXPECT_TRUE(collect_range(b, 40, 20).empty());
    EXPECT_EQ(collect_range(b, 30, 31), (std::vector<Key>{30}));
}

TEST(PmaEdge, EmptyBlock) {
    Block b = make_block({});
    EXPECT_TRUE(b.empty());
    EXPECT_FALSE(b.find(0).has_value());
    EXPECT_TRUE(collect(b).empty());
    b.insert(42, 42);
    b.append(43, 43);
    b.insert(1, 1);
    b.check_invariants();
    EXPECT_EQ(collect(b), (std::vector<Key>{1, 42, 43}));
}

TEST(PmaRankStability, RanksTrackSortedOrderAcrossRebalance) {
    Block b = make_block({});
    std::mt19937_64 rng(99);
    std::set<Key> ref;
    for (int i = 0; i < 2000; ++i) {
        Key k = rng() % 50000;
        if (ref.insert(k).second) b.insert(k, k);
    }
    b.check_invariants();
    Rank expected_rank = 0;
    for (Key k : ref) {
        auto slot = b.find(k);
        ASSERT_TRUE(slot.has_value());
        EXPECT_EQ(b.local_rank(*slot), expected_rank);
        ++expected_rank;
    }
}

TEST(PmaFindIn, FindsWithinFullWindow) {
    Block b = make_block({10, 20, 30, 40, 50});
    std::size_t lo = b.slot_of_rank(0);
    std::size_t hi = b.slot_of_rank(b.size() - 1);
    for (Key k : {10, 20, 30, 40, 50}) {
        auto s = b.find_in(k, lo, hi);
        ASSERT_TRUE(s.has_value());
        EXPECT_EQ(b.key_at(*s), k);
    }
    for (Key k : {5, 25, 55}) EXPECT_FALSE(b.find_in(k, lo, hi).has_value());
}

TEST(PmaFindIn, RespectsWindowBounds) {
    Block b = make_block({10, 20, 30, 40, 50});
    std::size_t lo = b.slot_of_rank(1), hi = b.slot_of_rank(3);
    EXPECT_TRUE(b.find_in(30, lo, hi).has_value());
    EXPECT_FALSE(b.find_in(10, lo, hi).has_value());
    EXPECT_FALSE(b.find_in(50, lo, hi).has_value());
}

TEST(PmaFindIn, SingleRankWindowInclusive) {
    Block b = make_block({10, 20, 30, 40, 50});
    std::size_t s2 = b.slot_of_rank(2);
    auto s = b.find_in(30, s2, s2);
    ASSERT_TRUE(s.has_value());
    EXPECT_EQ(b.key_at(*s), 30u);
    EXPECT_FALSE(b.find_in(20, s2, s2).has_value());
    EXPECT_FALSE(b.find_in(40, s2, s2).has_value());
}

TEST(PmaFindIn, WithGapsAfterInserts) {
    Block b = make_block({});
    for (Key k = 0; k < 500; ++k) b.insert(k * 2, k * 2);
    b.check_invariants();
    Rank r = 100;
    std::size_t lo = b.slot_of_rank(r - 3), hi = b.slot_of_rank(r + 3);
    auto s = b.find_in(200, lo, hi);
    ASSERT_TRUE(s.has_value());
    EXPECT_EQ(b.key_at(*s), 200u);
    EXPECT_FALSE(b.find_in(201, lo, hi).has_value());
    EXPECT_FALSE(b.find_in(220, lo, hi).has_value());
}

TEST(PmaFindInBound, WindowFlatAsWidthGrows) {
    const Rank eps_ceil = 2;
    const double rho_leaf = 0.10;
    const std::size_t rank_width = 2 * eps_ceil + 2;
    const double slot_bound = double(rank_width) / rho_leaf;

    for (std::size_t w : {std::size_t(1000), std::size_t(10000),
                          std::size_t(100000), std::size_t(1000000)}) {
        std::vector<Key> keys;
        keys.reserve(w);
        for (std::size_t j = 0; j < w; ++j) keys.push_back(Key(2 * j));
        std::vector<Rank> pl(w);
        for (std::size_t j = 0; j < w; ++j) pl[j] = Rank(j);
        Block b = Block::bulk_load(keys, pl, PmaBlock::capacity_for_key_cap(1u << 20));

        for (Rank pos : {Rank(0), Rank(w / 2), Rank(w - w / 10), Rank(w - 1)}) {
            const Rank lo_rank = (pos > eps_ceil) ? pos - eps_ceil : 0;
            Rank hi_rank = pos + eps_ceil + 1;
            if (hi_rank > Rank(w - 1)) hi_rank = Rank(w - 1);

            const std::size_t lo_slot = b.slot_of_rank(lo_rank);
            const std::size_t hi_slot = b.slot_of_rank(hi_rank);
            const std::size_t window = hi_slot - lo_slot + 1;

            EXPECT_LE(double(window), slot_bound)
                << "window blew up at w=" << w << " pos=" << pos;

            const Key target = b.key_at(b.slot_of_rank(pos));
            auto hit = b.find_in(target, lo_slot, hi_slot);
            ASSERT_TRUE(hit.has_value());
            EXPECT_EQ(b.key_at(*hit), target);
        }
    }
}

}