// test_segment_tree.cpp - differential harness for the mapping table B+tree:
// route/select/rank/edits vs flat reference vectors, verify() after every op,
// at B=8 for deep trees and B=32 as shipped

#include <gtest/gtest.h>

#include "cedar/segment_tree.hpp"

#include <vector>
#include <memory>
#include <random>
#include <algorithm>

using namespace li;

namespace {

using Item = std::unique_ptr<std::uint64_t>;

template <int B>
void run_churn(std::uint64_t seed, int rounds, int ops) {
    using Tree = li::detail::SegmentTree<Item, B>;
    std::mt19937_64 rng(seed);

    for (int round = 0; round < rounds; ++round) {
        Tree t;
        std::vector<Key> rkeys;
        std::vector<std::uint64_t> rvals;

        const std::size_t seed_n = rng() % 200;

        for (std::size_t i = 0; i < seed_n; ++i) {
            const Key k = rkeys.empty() ? 1000 + rng() % 100 : rkeys.back() + 1 + rng() % 100;

            rkeys.push_back(k);
            rvals.push_back(k * 7);
            t.insert_at(t.size(), k, std::make_unique<std::uint64_t>(k * 7));
        }

        t.check_invariants();

        for (int o = 0; o < ops; ++o) {
            const unsigned dice = unsigned(rng() % 100);

            if (dice < 45 || t.size() == 0) {
                const std::size_t i = rng() % (t.size() + 1);
                const Key lo = (i == 0) ? 0 : rkeys[i - 1];
                const Key hi = (i == rkeys.size()) ? lo + 1000 : rkeys[i];

                if (hi <= lo + 1) continue;

                const Key k = lo + 1 + rng() % (hi - lo - 1);

                t.insert_at(i, k, std::make_unique<std::uint64_t>(k * 7));
                rkeys.insert(rkeys.begin() + std::ptrdiff_t(i), k);
                rvals.insert(rvals.begin() + std::ptrdiff_t(i), k * 7);
            } else if (dice < 85) {
                const std::size_t i = rng() % t.size();
                Item out = t.erase_at(i);

                ASSERT_TRUE(out != nullptr);
                EXPECT_EQ(*out, rvals[i]);
                rkeys.erase(rkeys.begin() + std::ptrdiff_t(i));
                rvals.erase(rvals.begin() + std::ptrdiff_t(i));
            } else if (dice < 92 && t.size() > 0) {
                const std::size_t i = rng() % t.size();
                const Key lo = (i == 0) ? 0 : rkeys[i - 1];

                if (rkeys[i] <= lo + 1) continue;

                const Key k = lo + 1 + rng() % (rkeys[i] - lo - 1);

                t.set_sep(i, k);
                rkeys[i] = k;
            } else if (t.size() > 0) {
                const Key probe = rng() % (rkeys.back() + 100);
                std::size_t want = 0;

                for (std::size_t j = 0; j < rkeys.size(); ++j) {
                    if (rkeys[j] <= probe) want = j;
                }

                EXPECT_EQ(t.route(probe), want);

                const auto lc = t.route_located(probe);

                EXPECT_EQ(lc.index, want);
                ASSERT_TRUE(lc.item != nullptr && *lc.item != nullptr);
                EXPECT_EQ(*(*lc.item), rvals[want]);
            }

            t.check_invariants();
            ASSERT_EQ(t.size(), rkeys.size());

            if (t.size() > 0) {
                const std::size_t i = rng() % t.size();

                ASSERT_TRUE(t.at(i) != nullptr);
                EXPECT_EQ(*t.at(i), rvals[i]);

                if (i + 1 < t.size()) {
                    const auto pr = t.at_pair(i);

                    EXPECT_EQ(*(*pr.a), rvals[i]);
                    EXPECT_EQ(*(*pr.b), rvals[i + 1]);
                }
            }

            if (::testing::Test::HasFailure()) return;
        }

        std::vector<std::uint64_t> seen;
        t.for_each_item([&](const Item& it) { seen.push_back(*it); });

        EXPECT_EQ(seen, rvals);
    }
}

}

TEST(SegmentTree, EmptyContracts) {
    li::detail::SegmentTree<Item, 8> t;

    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0u);
    EXPECT_EQ(t.route(12345), 0u);

    const auto lc = t.route_located(12345);

    EXPECT_EQ(lc.index, 0u);
    EXPECT_EQ(lc.item, nullptr);

    t.check_invariants();
}

TEST(SegmentTree, ClearResetsToEmpty) {
    li::detail::SegmentTree<Item, 8> t;

    for (Key k = 0; k < 100; ++k) {
        t.insert_at(t.size(), 1000 + k, std::make_unique<std::uint64_t>(k));
    }

    t.clear();

    EXPECT_TRUE(t.empty());
    t.check_invariants();

    t.insert_at(0, 5, std::make_unique<std::uint64_t>(55));

    EXPECT_EQ(t.size(), 1u);
    EXPECT_EQ(*t.at(0), 55u);
}

TEST(SegmentTree, GrowThenDrainCollapsesRoot) {
    li::detail::SegmentTree<Item, 8> t;
    std::vector<std::uint64_t> vals;

    for (Key k = 0; k < 500; ++k) {
        t.insert_at(t.size(), 10 * k, std::make_unique<std::uint64_t>(k));
        vals.push_back(k);
    }

    t.check_invariants();

    std::mt19937_64 rng(3);

    while (t.size() > 1) {
        const std::size_t i = rng() % t.size();
        Item out = t.erase_at(i);

        ASSERT_TRUE(out != nullptr);
        EXPECT_EQ(*out, vals[i]);
        vals.erase(vals.begin() + std::ptrdiff_t(i));
        t.check_invariants();

        if (::testing::Test::HasFailure()) return;
    }

    EXPECT_EQ(t.size(), 1u);
    EXPECT_EQ(*t.at(0), vals[0]);
}

TEST(SegmentTree, ChurnDeepB8) {
    run_churn<8>(20260724, 80, 900);
}

TEST(SegmentTree, ChurnShippedB32) {
    run_churn<32>(20260725, 30, 900);
}
