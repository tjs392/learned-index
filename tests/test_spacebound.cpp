// testr_spacebound.cpp - independent brute force of the space bound
//
// check w irreducubility with minimal line cover
// uses an independent O(n^2) brute coverability solver instead and adds the check 
// the segment count bound M<= 2*m_{W_m}-1
// also churn

#include <gtest/gtest.h>

#include "li/cedar_index.hpp"

#include <vector>
#include <set>
#include <map>
#include <random>
#include <string>
#include <cstdint>

namespace {

using li::CedarIndex;
using li::Key;
using li::Payload;
using li::Status;

bool brute_coverable(const std::vector<Key>& keys, uint64_t a, uint64_t b, double eps) {
    double kl = double(keys[a]);
    double rho_lo = -1e300, rho_hi = 1e300;
    for (uint64_t i = a; i <= b; ++i) {
        for (uint64_t j = i + 1; j <= b; ++j) {
            double xi = double(keys[i]) - kl, yi = double(i - a);
            double xj = double(keys[j]) - kl, yj = double(j - a);
            double dx = xj - xi;
            if (dx <= 0) continue;
            double hi = ((yj + eps) - (yi - eps)) / dx;
            double lo = ((yj - eps) - (yi + eps)) / dx;
            if (hi < rho_hi) rho_hi = hi;
            if (lo > rho_lo) rho_lo = lo;
        }
    }
    return rho_lo <= rho_hi + 1e-12;
}

uint64_t brute_segment_count_capped(const std::vector<Key>& keys, double eps, uint64_t cap) {
    if (keys.empty()) return 0;
    uint64_t segs = 0, a = 0, n = keys.size();
    while (a < n) {
        uint64_t b = a;
        while (b + 1 < n && (b + 1 - a + 1) <= cap && brute_coverable(keys, a, b + 1, eps)) ++b;
        ++segs;
        a = b + 1;
    }
    return segs;
}

void check_space_bound(const CedarIndex& idx, const std::set<Key>& live, double eps) {
    const auto& segs = idx.segments_for_test();
    std::vector<Key> all(live.begin(), live.end());

    const uint64_t m_cap = brute_segment_count_capped(all, eps, idx.w_m());
    const uint64_t M = segs.size();

    if (m_cap == 0) {
        EXPECT_EQ(M, 0u) << "nonempty structure over an empty key set";
        return;
    }
    EXPECT_LE(M, 2 * m_cap - 1)
        << "L1' SPACE BOUND VIOLATED: M=" << M << " vs 2*m_{W_m}-1=" << (2 * m_cap - 1)
        << " (m_{W_m}=" << m_cap << ", n=" << all.size() << ", W_s=" << idx.w_s()
        << ", W_m=" << idx.w_m() << ")";

    for (std::size_t i = 0; i + 1 < segs.size(); ++i) {
        if (segs[i].count + segs[i + 1].count <= idx.w_m()) {
            std::vector<Key> u = segs[i].pma.dump_sorted().first;
            std::vector<Key> r = segs[i + 1].pma.dump_sorted().first;
            u.insert(u.end(), r.begin(), r.end());
            EXPECT_FALSE(brute_coverable(u, 0, u.size() - 1, eps))
                << "IRREDUCIBILITY VIOLATED (brute): segments " << i << " and " << (i + 1)
                << " coverable and under W_m but left unmerged (sizes "
                << segs[i].count << "+" << segs[i + 1].count << ")";
        }
    }
}

void run(double eps, std::size_t ws, std::size_t wm, int b, uint64_t seed, int ops, int nbase) {
    std::mt19937_64 rng(seed);
    std::vector<Key> base;
    uint64_t v = 0;
    for (int i = 0; i < nbase; ++i) { v += 1 + (rng() % 15); base.push_back(v); }

    std::set<Key> live;
    CedarIndex idx(eps, ws, wm, b);
    if (nbase > 0) {
        idx.build(base);
        for (Key k : base) live.insert(k);
    }
    check_space_bound(idx, live, eps);

    std::uniform_int_distribution<uint64_t> kd(1, v + 300);
    for (int n = 0; n < ops; ++n) {
        if ((rng() & 1) && !live.empty()) {
            auto it = live.begin();
            std::advance(it, rng() % live.size());
            Key k = *it;
            SCOPED_TRACE("ERASE " + std::to_string(k));
            ASSERT_EQ(idx.erase(k), Status::ok);
            live.erase(k);
        } else {
            Key k = kd(rng);
            while (live.count(k)) k = kd(rng);
            SCOPED_TRACE("INSERT " + std::to_string(k));
            ASSERT_EQ(idx.insert(k, Payload(n)), Status::ok);
            live.insert(k);
        }
        check_space_bound(idx, live, eps);
        if (::testing::Test::HasFailure()) return;
    }
}

TEST(SpaceBound, BigCapBuildAndChurn) {
    for (uint64_t s = 0; s < 4; ++s) {
        run(0.0, 2048, 1024, 11, 10 + s, 200, 60);
        run(0.5, 2048, 1024, 11, 20 + s, 200, 60);
        run(2.0, 2048, 1024, 11, 30 + s, 200, 50);
    }
}

TEST(SpaceBound, SmallCapForcesTiling) {
    for (uint64_t s = 0; s < 4; ++s) {
        run(1.0, 8, 4, 3, 40 + s, 200, 70);
        run(4.0, 8, 4, 3, 50 + s, 200, 70);
        run(2.0, 16, 8, 4, 60 + s, 200, 80);
    }
}

}