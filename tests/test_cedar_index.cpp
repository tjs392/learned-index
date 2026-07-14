// test_cedar_index.cpp - invariant + differential harness for CedarIndex.
//
// check_index verifies, after each op: count == pma.size(),
// count <= W_s, key_low strictly increasing, exact hard epsilon under the stored model, the band's
// min/max over occupied matches brute force, W-irreducibility (no adjacent pair both coverable and
// under W_m), no cascade (segment-count delta <= 2 per op), derived global rank, payload fidelity,
// and global sort order.

#include <gtest/gtest.h>

#include "li/cedar_index.hpp"
#include "li/minimal_line_cover.hpp"

#include <vector>
#include <set>
#include <map>
#include <random>
#include <cmath>

using namespace li;

namespace {

constexpr double kEpsSlack = 1e-9;
constexpr double kBandSlack = 1e-6;

void check_index(const CedarIndex& idx, const std::set<Key>& live,
                 const std::map<Key, Payload>& payloads) {
    const double eps = idx.epsilon();
    const auto& segs = idx.segments_for_test();

    if (live.empty()) {
        EXPECT_TRUE(segs.empty());
        return;
    }
    ASSERT_FALSE(segs.empty());

    std::size_t total = 0;
    std::vector<Key> in_order;
    for (std::size_t i = 0; i < segs.size(); ++i) {
        const Segment& s = segs[i];
        EXPECT_GT(s.count, 0u);
        EXPECT_EQ(s.count, s.pma.size());
        EXPECT_LE(s.count, idx.w_s());
        s.pma.check_invariants();
        if (i) { EXPECT_LT(segs[i - 1].key_low, s.key_low); }

        double band_min = 1e18;
        double band_max = -1e18;
        std::size_t local = 0;
        for (std::size_t slot = s.pma.next_occupied(0); slot < s.pma.capacity();
             slot = s.pma.next_occupied(slot + 1)) {
            Key k = s.pma.key_at(slot);
            double dev = s.model.alpha * double(k - s.key_low) + s.model.beta - double(local);

            EXPECT_LE(std::fabs(dev), eps + kEpsSlack) << "hard-eps seg " << i << " key " << k;

            if (dev < band_min) band_min = dev;
            if (dev > band_max) band_max = dev;
            in_order.push_back(k);
            ++local;
            ++total;
        }
        EXPECT_LE(std::fabs(band_min - s.tolerance_band.min_over_occupied()), kBandSlack)
            << "band min seg " << i;
        EXPECT_LE(std::fabs(band_max - s.tolerance_band.max_over_occupied()), kBandSlack)
            << "band max seg " << i;
    }

    EXPECT_EQ(total, live.size());
    EXPECT_EQ(std::vector<Key>(live.begin(), live.end()), in_order);

    for (std::size_t i = 0; i + 1 < segs.size(); ++i) {
        if (segs[i].count + segs[i + 1].count <= idx.w_m()) {
            auto [lk, lp] = segs[i].pma.dump_sorted();
            auto [rk, rp] = segs[i + 1].pma.dump_sorted();
            std::vector<Key> u(lk.begin(), lk.end());
            u.insert(u.end(), rk.begin(), rk.end());
            EXPECT_EQ(detail::minimal_line_cover(u, eps).status, detail::LineCoverStatus::SPLIT)
                << "adjacent coverable pair under W_m should have merged (" << i << "," << i + 1 << ")";
        }
    }

    Rank gi = 0;
    for (Key k : in_order) {
        auto pl = idx.point_lookup(k);
        EXPECT_TRUE(pl.ok());
        if (pl.ok()) { EXPECT_EQ(pl.value(), payloads.at(k)); }
        auto gr = idx.global_rank_of(k);
        EXPECT_TRUE(gr.ok());
        if (gr.ok()) { EXPECT_EQ(gr.value(), gi); }
        ++gi;
    }
}

void insert_and_check(CedarIndex& idx, Key k, Payload p,
                      std::set<Key>& live, std::map<Key, Payload>& payloads) {
    std::size_t before = idx.segments_for_test().size();
    EXPECT_EQ(idx.insert(k, p), Status::ok);
    live.insert(k);
    payloads[k] = p;
    EXPECT_LE(std::llabs((long long)idx.segments_for_test().size() - (long long)before), 2);
    check_index(idx, live, payloads);
}

void erase_and_check(CedarIndex& idx, Key k,
                     std::set<Key>& live, std::map<Key, Payload>& payloads) {
    std::size_t before = idx.segments_for_test().size();
    bool present = live.count(k) != 0;
    EXPECT_EQ(idx.erase(k) == Status::ok, present);
    if (present) { live.erase(k); payloads.erase(k); }
    EXPECT_LE(std::llabs((long long)idx.segments_for_test().size() - (long long)before), 2);
    check_index(idx, live, payloads);
}

CedarIndex build_index(const std::vector<Key>& keys, double eps,
                        std::size_t ws, std::size_t wm, int b,
                        std::set<Key>& live, std::map<Key, Payload>& payloads) {
    CedarIndex idx(eps, ws, wm, b);
    idx.build(keys);
    live.clear();
    payloads.clear();
    for (std::size_t i = 0; i < keys.size(); ++i) { live.insert(keys[i]); payloads[keys[i]] = Payload(i); }
    return idx;
}

void run_churn(double eps, std::size_t ws, std::size_t wm, int b,
               uint64_t seed, int ops, int nbase) {
    std::mt19937_64 rng(seed);
    std::vector<Key> base;
    uint64_t v = 0;
    for (int i = 0; i < nbase; ++i) { v += 1 + (rng() % 20); base.push_back(v); }

    std::set<Key> live;
    std::map<Key, Payload> pay;
    CedarIndex idx = build_index(base, eps, ws, wm, b, live, pay);
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
    std::set<Key> live; std::map<Key, Payload> pay;
    std::vector<Key> k = {0, 10, 20, 30, 40};
    CedarIndex idx = build_index(k, 0.0, 2048, 1024, 11, live, pay);
    insert_and_check(idx, 15, 424242, live, pay);
    EXPECT_EQ(idx.segments_for_test().size(), 3u);
}

TEST(CedarIndex, FrontInsertLowersKeyLow) {
    std::set<Key> live; std::map<Key, Payload> pay;
    std::vector<Key> k;
    for (uint64_t i = 0; i < 20; ++i) k.push_back(100 + 10 * i);
    CedarIndex idx = build_index(k, 2.0, 2048, 1024, 11, live, pay);
    insert_and_check(idx, 50, 314159, live, pay);
    EXPECT_EQ(idx.segments_for_test().front().key_low, 50u);
    EXPECT_EQ(idx.global_rank_of(50).value(), 0u);
    EXPECT_EQ(idx.global_rank_of(100).value(), 1u);
}

TEST(CedarIndex, PayloadsFrozenRanksDerived) {
    std::set<Key> live; std::map<Key, Payload> pay;
    std::vector<Key> k = {10, 20, 30, 40, 50};
    CedarIndex idx = build_index(k, 2.0, 2048, 1024, 11, live, pay);
    insert_and_check(idx, 25, 999, live, pay);
    EXPECT_EQ(idx.point_lookup(25).value(), 999u);
    EXPECT_EQ(idx.point_lookup(30).value(), 2u);
    EXPECT_EQ(idx.global_rank_of(30).value(), 3u);
}

TEST(CedarIndex, EraseEmptiesAndMerges) {
    std::set<Key> live; std::map<Key, Payload> pay;
    std::vector<Key> k = {10, 20, 30};
    CedarIndex idx = build_index(k, 0.0, 2048, 1024, 11, live, pay);
    erase_and_check(idx, 20, live, pay);
    erase_and_check(idx, 10, live, pay);
    erase_and_check(idx, 30, live, pay);
    EXPECT_TRUE(live.empty());
    EXPECT_TRUE(idx.segments_for_test().empty());
}

TEST(CedarIndex, ChurnBigCap) {
    for (uint64_t s = 0; s < 4; ++s) {
        run_churn(0.5, 2048, 1024, 11, 100 + s, 500, 80);
        run_churn(1.0, 2048, 1024, 11, 200 + s, 500, 80);
        run_churn(2.0, 2048, 1024, 11, 300 + s, 500, 80);
        run_churn(8.0, 2048, 1024, 11, 400 + s, 500, 60);
    }
}

TEST(CedarIndex, ChurnSmallCapForcesSplitsAndMerges) {
    for (uint64_t s = 0; s < 4; ++s) {
        run_churn(1.0, 8, 4, 3, 500 + s, 900, 0);
        run_churn(4.0, 8, 4, 3, 600 + s, 900, 0);
        run_churn(16.0, 16, 8, 4, 700 + s, 900, 0);
    }
}

TEST(CedarIndex, BuildFromEmptyExactEps) {
    for (uint64_t s = 0; s < 4; ++s)
        run_churn(0.0, 2048, 1024, 11, 800 + s, 300, 0);
}

TEST(CedarIndex, FrontInsertRebaseAvoidsRebuildStorm) {
    std::set<Key> live; std::map<Key, Payload> pay;
    std::vector<Key> seed;
    for (uint64_t i = 0; i < 10; ++i) seed.push_back(100000 + i);
    CedarIndex idx = build_index(seed, 0.0, 2048, 1024, 11, live, pay);

    const int count = 1000;
    for (int i = 0; i < count; ++i) {
        Key k = 100000 - 1 - uint64_t(i);
        insert_and_check(idx, k, Payload(500000000ull + uint64_t(i)), live, pay);
        if (::testing::Test::HasFailure()) return;
    }
    // Fixed: 0 rebuilds. Unfixed: one per insert (== count). Slack of 2 for paranoia.
    EXPECT_LE(idx.cover_recomputes(), 2u)
        << "front-insert re-base regressed: rebuild firing per new-minimum insert";
}