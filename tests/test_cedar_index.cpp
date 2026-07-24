// test_cedar_index.cpp - invariant + differential harness

#include <gtest/gtest.h>

#include "cedar/cedar_index.hpp"
#include "cedar/segmentation.hpp"

#include <vector>
#include <set>
#include <map>
#include <random>
#include <cmath>

using namespace li;

namespace {

constexpr double kEpsSlack = 1e-9;
constexpr double kGuardSlack = 1e-6;

void check_index(const CedarIndex& idx, const std::set<Key>& live,
                 const std::map<Key, Payload>& payloads) {
    const double eps = idx.epsilon();

    if (live.empty()) {
        EXPECT_EQ(idx.table_size_for_test(), 0u);
        return;
    }

    ASSERT_GT(idx.table_size_for_test(), 0u);

    std::size_t total = 0;
    std::vector<Key> in_order;
    std::vector<const Segment*> segs;

    idx.for_each_segment_for_test([&](const Segment& s) { segs.push_back(&s); });

    ASSERT_EQ(segs.size(), idx.table_size_for_test());

    for (std::size_t i = 0; i < segs.size(); ++i) {
        const Segment& s = *segs[i];

        EXPECT_GT(s.count, 0u);
        EXPECT_EQ(s.count, s.store.size());
        EXPECT_LE(s.count, idx.w_s());
        s.store.check_invariants();

        if (i) {
            EXPECT_LT(segs[i - 1]->key_low, s.key_low);
        }

        EXPECT_LE(s.key_low, s.store.keys_view()[0]);

        double band_min = 1e18;
        double band_max = -1e18;
        std::size_t local = 0;

        for (Key k : s.store.keys_view()) {
            const double dev = line_at(s.model, k, s.key_low) - double(local);

            EXPECT_LE(std::fabs(dev), eps + kEpsSlack) << "hard-eps seg " << i << " key " << k;

            band_min = std::min(band_min, dev);
            band_max = std::max(band_max, dev);
            in_order.push_back(k);
            ++local;
            ++total;
        }

        EXPECT_LE(s.guard.deviation_floor(), band_min + kGuardSlack)
            << "guard floor above the true min deviation, seg " << i << " (UNSOUND)";
        EXPECT_GE(s.guard.deviation_ceiling(), band_max - kGuardSlack)
            << "guard ceiling below the true max deviation, seg " << i << " (UNSOUND)";
    }

    EXPECT_EQ(total, live.size());
    EXPECT_EQ(std::vector<Key>(live.begin(), live.end()), in_order);

    for (std::size_t i = 0; i + 1 < segs.size(); ++i) {
        if (segs[i]->count + segs[i + 1]->count <= idx.w_m()) {
            std::span<const Key> lk = segs[i]->store.keys_view();
            std::span<const Key> rk = segs[i + 1]->store.keys_view();
            std::vector<Key> u(lk.begin(), lk.end());

            u.insert(u.end(), rk.begin(), rk.end());

            EXPECT_EQ(detail::minimal_line_cover(u, idx.epsilon_fit()).status,
                      detail::LineCoverStatus::SPLIT)
                << "adjacent coverable pair under w_m should have merged (" << i << "," << i + 1 << ")";
        }
    }

    for (Key k : in_order) {
        const auto pl = idx.point_lookup(k);

        ASSERT_TRUE(pl.ok()) << "live key " << k << " not found";
        EXPECT_EQ(pl.value(), payloads.at(k));
    }

    std::vector<Key> ranged;
    idx.range_scan(in_order.front(), in_order.back(), [&](Key k) { ranged.push_back(k); });

    EXPECT_EQ(ranged, in_order);
}

void insert_and_check(CedarIndex& idx, Key k, Payload p,
                      std::set<Key>& live, std::map<Key, Payload>& payloads) {
    const long long before = (long long)idx.table_size_for_test();

    EXPECT_EQ(idx.insert(k, p), Status::ok);
    live.insert(k);
    payloads[k] = p;

    EXPECT_LE(std::llabs((long long)idx.table_size_for_test() - before), 5);
    check_index(idx, live, payloads);
}

void erase_and_check(CedarIndex& idx, Key k,
                     std::set<Key>& live, std::map<Key, Payload>& payloads) {
    const long long before = (long long)idx.table_size_for_test();
    const bool present = live.count(k) != 0;

    EXPECT_EQ(idx.erase(k) == Status::ok, present);

    if (present) {
        live.erase(k);
        payloads.erase(k);
    }

    EXPECT_LE(std::llabs((long long)idx.table_size_for_test() - before), 5);
    check_index(idx, live, payloads);
}

void build_index(CedarIndex& idx, const std::vector<Key>& keys,
                 std::set<Key>& live, std::map<Key, Payload>& payloads) {
    idx.set_merge_prop(0);
    idx.build(keys);
    live.clear();
    payloads.clear();

    for (std::size_t i = 0; i < keys.size(); ++i) {
        live.insert(keys[i]);
        payloads[keys[i]] = Payload(i);
    }
}

void run_churn(double eps, std::size_t ws, std::size_t wm, int b,
               uint64_t seed, int ops, int nbase) {
    std::mt19937_64 rng(seed);
    std::vector<Key> base;
    uint64_t v = 0;

    for (int i = 0; i < nbase; ++i) {
        v += 1 + (rng() % 20);
        base.push_back(v);
    }

    std::set<Key> live;
    std::map<Key, Payload> pay;
    CedarIndex idx(eps, ws, wm, b);

    build_index(idx, base, live, pay);
    check_index(idx, live, pay);

    std::uniform_int_distribution<uint64_t> kd(1, v + 2000);

    for (int n = 0; n < ops; ++n) {
        if ((rng() & 1) == 0 && !live.empty()) {
            auto it = live.begin();

            std::advance(it, rng() % live.size());
            erase_and_check(idx, *it, live, pay);
        } else {
            Key k = kd(rng);

            while (live.count(k)) k = kd(rng);

            insert_and_check(idx, k, Payload(1000000000ull + uint64_t(n)), live, pay);
        }

        if (::testing::Test::HasFailure()) return;
    }
}

}

TEST(CedarIndex, OffTrendInsertForcesSplit) {
    std::set<Key> live;
    std::map<Key, Payload> pay;
    std::vector<Key> k = { 0, 10, 20, 30, 40 };
    CedarIndex idx(0.0, 2048, 1024, 11);

    build_index(idx, k, live, pay);

    insert_and_check(idx, 15, 424242, live, pay);

    EXPECT_GT(idx.table_size_for_test(), 1u);
}

TEST(CedarIndex, FrontInsertLowersKeyLow) {
    std::set<Key> live;
    std::map<Key, Payload> pay;
    std::vector<Key> k;

    for (uint64_t i = 0; i < 20; ++i) k.push_back(100 + 10 * i);

    CedarIndex idx(2.0, 2048, 1024, 11);

    build_index(idx, k, live, pay);

    insert_and_check(idx, 50, 314159, live, pay);

    Key first_low = 0;
    bool got_first = false;

    idx.for_each_segment_for_test([&](const Segment& s) {
        if (!got_first) {
            first_low = s.key_low;
            got_first = true;
        }
    });

    ASSERT_TRUE(got_first);
    EXPECT_EQ(first_low, 50u);
    EXPECT_EQ(idx.point_lookup(50).value(), 314159u);
}

TEST(CedarIndex, PayloadsAreFrozen) {
    std::set<Key> live;
    std::map<Key, Payload> pay;
    std::vector<Key> k = { 10, 20, 30, 40, 50 };
    CedarIndex idx(2.0, 2048, 1024, 11);

    build_index(idx, k, live, pay);

    insert_and_check(idx, 25, 999, live, pay);

    EXPECT_EQ(idx.point_lookup(25).value(), 999u);
    EXPECT_EQ(idx.point_lookup(30).value(), 2u);
}

TEST(CedarIndex, AbsentKeyContract) {
    std::set<Key> live;
    std::map<Key, Payload> pay;
    std::vector<Key> k = { 10, 20, 30 };
    CedarIndex idx(2.0, 2048, 1024, 11);

    build_index(idx, k, live, pay);

    EXPECT_EQ(idx.point_lookup(15).status(), Status::not_found);
    EXPECT_EQ(idx.point_lookup(5).status(), Status::not_found);
    EXPECT_EQ(idx.point_lookup(500).status(), Status::not_found);
    EXPECT_EQ(idx.erase(15), Status::not_found);

    check_index(idx, live, pay);
}

TEST(CedarIndex, EraseEmptiesTheIndex) {
    std::set<Key> live;
    std::map<Key, Payload> pay;
    std::vector<Key> k = { 10, 20, 30 };
    CedarIndex idx(0.0, 2048, 1024, 11);

    build_index(idx, k, live, pay);

    erase_and_check(idx, 20, live, pay);
    erase_and_check(idx, 10, live, pay);
    erase_and_check(idx, 30, live, pay);

    EXPECT_TRUE(live.empty());
    EXPECT_EQ(idx.table_size_for_test(), 0u);
    EXPECT_EQ(idx.point_lookup(20).status(), Status::not_found);
}

TEST(CedarIndex, RangeScanMatchesReference) {
    std::set<Key> live;
    std::map<Key, Payload> pay;
    std::mt19937_64 rng(9);
    std::vector<Key> k;
    Key v = 100;

    for (int i = 0; i < 400; ++i) {
        k.push_back(v);
        v += 1 + rng() % 25;
    }

    CedarIndex idx(4.0, 2048, 1024, 11);

    build_index(idx, k, live, pay);

    for (int q = 0; q < 200; ++q) {
        const Key lo = rng() % (v + 100);
        const Key hi = lo + rng() % 2000;

        std::vector<Key> got;
        idx.range_scan(lo, hi, [&](Key kk) { got.push_back(kk); });

        std::vector<Key> want;

        for (auto it = live.lower_bound(lo); it != live.end() && *it <= hi; ++it) {
            want.push_back(*it);
        }

        EXPECT_EQ(got, want);

        if (::testing::Test::HasFailure()) return;
    }
}

TEST(CedarIndex, ChurnBigCap) {
    for (uint64_t s = 0; s < 3; ++s) {
        run_churn(0.5, 2048, 1024, 11, 100 + s, 400, 80);
        run_churn(1.0, 2048, 1024, 11, 200 + s, 400, 80);
        run_churn(2.0, 2048, 1024, 11, 300 + s, 400, 80);
        run_churn(8.0, 2048, 1024, 11, 400 + s, 400, 60);
    }
}

TEST(CedarIndex, ChurnSmallCapForcesSplitsAndMerges) {
    for (uint64_t s = 0; s < 3; ++s) {
        run_churn(1.0, 8, 4, 3, 500 + s, 700, 0);
        run_churn(4.0, 8, 4, 3, 600 + s, 700, 0);
        run_churn(16.0, 16, 8, 4, 700 + s, 700, 0);
    }
}

TEST(CedarIndex, BuildFromEmptyExactEps) {
    for (uint64_t s = 0; s < 3; ++s) {
        run_churn(0.0, 2048, 1024, 11, 800 + s, 250, 0);
    }
}

TEST(CedarIndex, FrontInsertRebaseAvoidsRebuildStorm) {
    std::set<Key> live;
    std::map<Key, Payload> pay;
    std::vector<Key> seed;

    for (uint64_t i = 0; i < 10; ++i) seed.push_back(100000 + i);

    CedarIndex idx(0.0, 2048, 1024, 11);

    build_index(idx, seed, live, pay);

    const int count = 1000;

    for (int i = 0; i < count; ++i) {
        const Key k = 100000 - 1 - uint64_t(i);

        insert_and_check(idx, k, Payload(500000000ull + uint64_t(i)), live, pay);

        if (::testing::Test::HasFailure()) return;
    }

    EXPECT_LE(idx.cover_recomputes(), 2u)
        << "front-insert re-base regressed: rebuild firing per new-minimum insert";
}
