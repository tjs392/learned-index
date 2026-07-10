#include <gtest/gtest.h>
#include "li/hull_tree.hpp"

#include <vector>
#include <set>
#include <random>
#include <algorithm>
#include <iterator>
#include <cstdint>

namespace {

using li::detail::HullTree;
using li::Key;
using li::Rank;

std::vector<Key> in_order_keys(const HullTree& t) {
    std::vector<Key> out;
    t.for_each([&](Key k) { out.push_back(k); });
    return out;
}

void check_against(const HullTree& t, const std::set<Key>& ref) {
    t.validate();
    ASSERT_EQ(t.size(), ref.size());

    std::vector<Key> sorted(ref.begin(), ref.end());
    EXPECT_EQ(in_order_keys(t), sorted);

    for (Rank r = 0; r < sorted.size(); ++r) {
        EXPECT_EQ(t.select(r), sorted[r]) << "select mismatch at r=" << r;
        EXPECT_TRUE(t.contains(sorted[r]));
        EXPECT_EQ(t.rank_of(sorted[r]), r) << "rank mismatch for present key " << sorted[r];
    }
}

Rank ref_rank(const std::set<Key>& ref, Key k) {
    return static_cast<Rank>(std::distance(ref.begin(), ref.lower_bound(k)));
}

TEST(HullTreeBasic, EmptyAndSingle) {
    HullTree t;
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0u);
    EXPECT_FALSE(t.contains(5));
    EXPECT_EQ(t.rank_of(5), 0u);

    t.insert(42);
    EXPECT_EQ(t.size(), 1u);
    EXPECT_TRUE(t.contains(42));
    EXPECT_FALSE(t.contains(41));
    EXPECT_EQ(t.rank_of(10), 0u);
    EXPECT_EQ(t.rank_of(42), 0u);
    EXPECT_EQ(t.rank_of(100), 1u);
    EXPECT_EQ(t.select(0), 42u);

    EXPECT_TRUE(t.erase(42));
    EXPECT_TRUE(t.empty());
    EXPECT_FALSE(t.erase(42));
}

TEST(HullTreeBasic, AscendingInsertsForceRebuilds) {
    HullTree t;
    std::set<Key> ref;
    for (Key k = 0; k < 500; ++k) {
        t.insert(k * 3);
        ref.insert(k * 3);
        t.validate();
    }
    check_against(t, ref);
}

TEST(HullTreeBasic, DescendingInserts) {
    HullTree t;
    std::set<Key> ref;
    for (Key k = 500; k > 0; --k) {
        t.insert(k * 2);
        ref.insert(k * 2);
        t.validate();
    }
    check_against(t, ref);
}

TEST(HullTreeBasic, RankAbsentKeys) {
    HullTree t;
    std::set<Key> ref;
    for (Key k = 0; k < 100; ++k) { t.insert(k * 10); ref.insert(k * 10); }
    for (Key q : {Key(0), Key(5), Key(15), Key(500), Key(995), Key(1000), Key(99999)}) {
        EXPECT_EQ(t.rank_of(q), ref_rank(ref, q)) << "absent-rank mismatch for " << q;
    }
}

TEST(HullTreeBulk, BuildMatchesInserts) {
    std::vector<Key> keys;
    for (Key k = 0; k < 1000; ++k) keys.push_back(k * 7 + 3);
    HullTree t = HullTree::bulk_build(keys);
    std::set<Key> ref(keys.begin(), keys.end());
    check_against(t, ref);
}

TEST(HullTreeChurn, RandomInsertErase) {
    std::mt19937_64 rng(20260708);
    HullTree t;
    std::set<Key> ref;
    std::uniform_int_distribution<Key> kd(0, 20000);

    for (int step = 0; step < 20000; ++step) {
        const bool ins = ref.empty() || (rng() & 1);
        if (ins) {
            Key k = kd(rng);
            while (ref.count(k)) k = kd(rng);
            t.insert(k);
            ref.insert(k);
        } else {
            std::uniform_int_distribution<std::size_t> pick(0, ref.size() - 1);
            auto it = ref.begin();
            std::advance(it, static_cast<std::ptrdiff_t>(pick(rng)));
            Key k = *it;
            EXPECT_TRUE(t.erase(k));
            ref.erase(it);
        }
        if ((step % 250) == 0) check_against(t, ref);
    }
    check_against(t, ref);
}

TEST(HullTreeChurn, BulkThenChurn) {
    std::vector<Key> keys;
    for (Key k = 0; k < 800; ++k) keys.push_back(k * 5);
    HullTree t = HullTree::bulk_build(keys);
    std::set<Key> ref(keys.begin(), keys.end());

    std::mt19937_64 rng(999);
    std::uniform_int_distribution<Key> kd(0, 5000);
    for (int step = 0; step < 8000; ++step) {
        const bool ins = ref.empty() || (rng() % 3 != 0);
        if (ins) {
            Key k = kd(rng);
            while (ref.count(k)) k = kd(rng);
            t.insert(k);
            ref.insert(k);
        } else {
            std::uniform_int_distribution<std::size_t> pick(0, ref.size() - 1);
            auto it = ref.begin();
            std::advance(it, static_cast<std::ptrdiff_t>(pick(rng)));
            EXPECT_TRUE(t.erase(*it));
            ref.erase(it);
        }
        if ((step % 200) == 0) check_against(t, ref);
    }
    check_against(t, ref);
}

}