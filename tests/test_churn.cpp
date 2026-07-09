#include <gtest/gtest.h>
#include "li/index.hpp"
#include "li/status.hpp"
#include "li/model.hpp"

#include <vector>
#include <set>
#include <map>
#include <random>
#include <string>
#include <cmath>
#include <cstdint>
#include <iterator>

namespace {

using li::LearnedIndex;
using li::Status;
using li::Key;
using li::Payload;
using li::Rank;

constexpr double kEpsSlack = 1e-9;

bool brute_coverable(const std::vector<Key>& keys, uint64_t a, uint64_t b, double eps) {
    double kl = static_cast<double>(keys[a]);
    double rho_lo = -1e300, rho_hi = 1e300;
    for (uint64_t i = a; i <= b; ++i) {
        for (uint64_t j = i + 1; j <= b; ++j) {
            double xi = static_cast<double>(keys[i]) - kl, yi = static_cast<double>(i - a);
            double xj = static_cast<double>(keys[j]) - kl, yj = static_cast<double>(j - a);
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

uint64_t brute_segment_count(const std::vector<Key>& keys, double eps) {
    if (keys.empty()) return 0;
    uint64_t segs = 0, a = 0, n = keys.size();
    while (a < n) {
        uint64_t b = a;
        while (b + 1 < n && brute_coverable(keys, a, b + 1, eps)) ++b;
        ++segs;
        a = b + 1;
    }
    return segs;
}

void check_index(const LearnedIndex& idx,
                 const std::set<Key>& live,
                 const std::map<Key, Payload>& payloads,
                 double eps) {
    const auto& mt = idx.mapping_table_for_test();
    const auto& bl = idx.blocks_for_test();
    ASSERT_EQ(mt.size(), bl.size()) << "descriptor/block vectors misaligned";

    if (live.empty()) { EXPECT_TRUE(mt.empty()); return; }
    ASSERT_FALSE(mt.empty());

    std::vector<Rank> base(mt.size(), 0);
    Rank acc = 0;
    for (size_t i = 0; i < mt.size(); ++i) {
        base[i] = acc;
        EXPECT_GT(mt[i].count, 0u) << "empty segment at " << i;
        EXPECT_EQ(mt[i].count, bl[i].size()) << "count != block size at " << i;
        if (i) {
            EXPECT_LT(mt[i - 1].key_low, mt[i].key_low)
                << "key_low not strictly increasing at " << i;
        }
        bl[i].check_invariants();
        acc += mt[i].count;
    }
    EXPECT_EQ(acc, live.size()) << "sum of counts != live key count";

    Rank gi = 0;
    for (Key k : live) {
        const size_t i = idx.find_descriptor(k);
        const auto& d = mt[i];
        EXPECT_LE(d.key_low, k) << "key " << k << " below its segment key_low";
        if (i + 1 < mt.size()) {
            EXPECT_LT(k, mt[i + 1].key_low) << "key " << k << " leaked past its segment";
        }
        const Rank local = gi - base[i];
        const double pred = d.model.alpha * double(k - d.key_low) + d.model.beta;
        EXPECT_LE(std::fabs(pred - double(local)), eps + kEpsSlack)
            << "HARD-EPS VIOLATION key " << k << " seg " << i
            << " local_rank " << local << " pred " << pred << " eps " << eps;

        auto gr = idx.global_rank_of(k);
        ASSERT_TRUE(gr.ok()) << "global_rank_of missing live key " << k;
        EXPECT_EQ(gr.value(), gi) << "wrong derived rank for key " << k;

        auto pl = idx.point_lookup(k);
        ASSERT_TRUE(pl.ok()) << "live key " << k << " not found";
        EXPECT_EQ(pl.value(), payloads.at(k)) << "wrong payload for key " << k;
        ++gi;
    }

    std::vector<Key> scanned;
    for (Key k : idx.range_lookup(*live.begin(), *live.rbegin())) scanned.push_back(k);
    EXPECT_EQ(scanned, std::vector<Key>(live.begin(), live.end()))
        << "range scan mismatch after structural change";
}

void check_count_bound_and_irreducibility(const LearnedIndex& idx,
                                          const std::set<Key>& live,
                                          double eps) {
    const auto& mt = idx.mapping_table_for_test();
    const auto& bl = idx.blocks_for_test();

    std::vector<Key> all(live.begin(), live.end());
    const uint64_t m_opt = brute_segment_count(all, eps);
    const uint64_t M = mt.size();

    if (m_opt == 0) {
        EXPECT_EQ(M, 0u) << "nonempty structure over an empty key set";
    } else {
        EXPECT_LE(M, 2 * m_opt - 1)
            << "COUNT BOUND VIOLATED: M=" << M << " vs 2*m_opt-1=" << (2 * m_opt - 1)
            << " (m_opt=" << m_opt << ", live=" << live.size() << ")";
    }

    for (size_t i = 0; i + 1 < mt.size(); ++i) {
        std::vector<Key> u = bl[i].dump_sorted().first;
        std::vector<Key> right = bl[i + 1].dump_sorted().first;
        u.insert(u.end(), right.begin(), right.end());
        const bool coverable = brute_coverable(u, 0, u.size() - 1, eps);
        EXPECT_FALSE(coverable)
            << "IRREDUCIBILITY VIOLATED: segments " << i << " and " << (i + 1)
            << " are jointly coverable but were left unmerged"
            << " (sizes " << bl[i].size() << "+" << bl[i + 1].size() << ")";
    }
}

void step_insert(LearnedIndex& idx, std::set<Key>& live, std::map<Key, Payload>& payloads,
                 double eps, Key k, Payload p) {
    SCOPED_TRACE("INSERT " + std::to_string(k));
    ASSERT_EQ(live.count(k), 0u) << "test bug: duplicate insert";
    ASSERT_EQ(idx.insert(k, p), Status::ok);
    live.insert(k);
    payloads[k] = p;
    check_index(idx, live, payloads, eps);
    check_count_bound_and_irreducibility(idx, live, eps);
}

void step_erase(LearnedIndex& idx, std::set<Key>& live, std::map<Key, Payload>& payloads,
                double eps, Key k) {
    SCOPED_TRACE("ERASE " + std::to_string(k));
    ASSERT_EQ(live.count(k), 1u) << "test bug: erase of absent key";
    ASSERT_EQ(idx.erase(k), Status::ok);
    live.erase(k);
    payloads.erase(k);
    check_index(idx, live, payloads, eps);
    check_count_bound_and_irreducibility(idx, live, eps);
}

Key pick_present(const std::set<Key>& live, std::mt19937_64& rng) {
    std::uniform_int_distribution<size_t> d(0, live.size() - 1);
    auto it = live.begin();
    std::advance(it, static_cast<std::ptrdiff_t>(d(rng)));
    return *it;
}

Key pick_absent(const std::set<Key>& live, std::mt19937_64& rng, uint64_t lo, uint64_t hi) {
    std::uniform_int_distribution<uint64_t> d(lo, hi);
    Key k = d(rng);
    while (live.count(k)) k = d(rng);
    return k;
}

LearnedIndex build_index(const std::vector<Key>& sorted_keys, double eps,
                         std::set<Key>& live, std::map<Key, Payload>& payloads) {
    LearnedIndex idx(eps);
    idx.build(sorted_keys);
    live.clear();
    payloads.clear();
    for (uint64_t i = 0; i < sorted_keys.size(); ++i) {
        live.insert(sorted_keys[i]);
        payloads[sorted_keys[i]] = static_cast<Payload>(i);
    }
    return idx;
}

std::vector<Key> linear_gap_keys(std::mt19937_64& rng, uint64_t n, uint64_t max_gap) {
    std::vector<Key> keys;
    keys.reserve(n);
    uint64_t v = 0;
    for (uint64_t i = 0; i < n; ++i) { v += 1 + (rng() % max_gap); keys.push_back(v); }
    return keys;
}

TEST(Churn, MixedInsertErase) {
    for (double eps : {0.0, 0.5, 1.0, 2.0, 8.0}) {
        std::mt19937_64 rng(0xC0FFEE + uint64_t(eps * 10));
        std::set<Key> live; std::map<Key, Payload> payloads;
        LearnedIndex idx = build_index(linear_gap_keys(rng, 60, 20), eps, live, payloads);
        check_index(idx, live, payloads, eps);
        check_count_bound_and_irreducibility(idx, live, eps);

        Payload pc = 1;
        for (int step = 0; step < 400; ++step) {
            const bool do_ins = live.empty() || (rng() & 1);
            if (do_ins) step_insert(idx, live, payloads, eps, pick_absent(live, rng, 1, 4000), pc++);
            else        step_erase(idx, live, payloads, eps, pick_present(live, rng));
            if (::testing::Test::HasFailure()) return;
        }
    }
}

TEST(Churn, DeleteHeavyShrinkage) {
    for (double eps : {0.5, 1.0, 4.0}) {
        std::mt19937_64 rng(0xDEAD + uint64_t(eps * 10));
        std::set<Key> live; std::map<Key, Payload> payloads;
        LearnedIndex idx = build_index(linear_gap_keys(rng, 150, 12), eps, live, payloads);

        Payload pc = 1;
        for (int step = 0; step < 400 && !live.empty(); ++step) {
            const bool do_ins = (rng() % 4) == 0;
            if (do_ins) step_insert(idx, live, payloads, eps, pick_absent(live, rng, 1, 4000), pc++);
            else        step_erase(idx, live, payloads, eps, pick_present(live, rng));
            if (::testing::Test::HasFailure()) return;
        }
    }
}

TEST(Churn, OscillatingSplitMerge) {
    const double eps = 0.5;
    std::set<Key> live; std::map<Key, Payload> payloads;
    std::vector<Key> base;
    for (uint64_t i = 0; i < 21; ++i) base.push_back(i * 100);
    LearnedIndex idx = build_index(base, eps, live, payloads);
    check_count_bound_and_irreducibility(idx, live, eps);

    Payload pc = 1;
    for (int cycle = 0; cycle < 150; ++cycle) {
        const Key off = 50 + 100 * (cycle % 20);
        step_insert(idx, live, payloads, eps, off, pc++);
        if (::testing::Test::HasFailure()) return;
        step_erase(idx, live, payloads, eps, off);
        if (::testing::Test::HasFailure()) return;
    }
    EXPECT_EQ(idx.mapping_table_for_test().size(), 1u)
        << "oscillation drifted: coverable set left as " << idx.mapping_table_for_test().size()
        << " segments";
}

TEST(Churn, FrontAndBackChurn) {
    for (double eps : {0.5, 2.0}) {
        std::mt19937_64 rng(0xF00D + uint64_t(eps * 10));
        std::set<Key> live; std::map<Key, Payload> payloads;
        LearnedIndex idx = build_index(linear_gap_keys(rng, 80, 10), eps, live, payloads);

        Payload pc = 1;
        for (int step = 0; step < 200; ++step) {
            const Key front = *live.begin();
            const Key back  = *live.rbegin();
            switch (rng() % 4) {
                case 0: step_insert(idx, live, payloads, eps, front > 1 ? front - 1 : 0, pc++); break;
                case 1: step_insert(idx, live, payloads, eps, back + 1 + (rng() % 50), pc++);    break;
                case 2: if (live.size() > 1) step_erase(idx, live, payloads, eps, front);        break;
                default: if (live.size() > 1) step_erase(idx, live, payloads, eps, back);        break;
            }
            if (::testing::Test::HasFailure()) return;
        }
    }
}

TEST(Churn, DeleteToEmptyThenRebuild) {
    const double eps = 1.0;
    std::mt19937_64 rng(0x1234);
    std::set<Key> live; std::map<Key, Payload> payloads;
    LearnedIndex idx = build_index(linear_gap_keys(rng, 100, 15), eps, live, payloads);

    while (!live.empty()) {
        step_erase(idx, live, payloads, eps, pick_present(live, rng));
        if (::testing::Test::HasFailure()) return;
    }
    EXPECT_TRUE(idx.mapping_table_for_test().empty()) << "structure not empty after deleting all keys";

    Payload pc = 1;
    for (int step = 0; step < 120; ++step) {
        step_insert(idx, live, payloads, eps, pick_absent(live, rng, 1, 5000), pc++);
        if (::testing::Test::HasFailure()) return;
    }
    EXPECT_GE(idx.mapping_table_for_test().size(), 1u);
}

}