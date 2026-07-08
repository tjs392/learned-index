#include <gtest/gtest.h>
#include "li/segmentation.hpp"
#include "li/model.hpp"

#include <vector>
#include <cmath>
#include <cstdint>
#include <random>

namespace {

using li::detail::FittedSegment;
using li::detail::segment_stream;

std::vector<li::Key> random_keys(std::mt19937_64& rng, uint64_t n, uint64_t max_gap) {
    std::vector<li::Key> keys;
    keys.reserve(n);
    std::uniform_int_distribution<uint64_t> gap(1, max_gap);
    uint64_t k = gap(rng);
    for (uint64_t i = 0; i < n; ++i) {
        keys.push_back(k);
        k += gap(rng);
    }
    return keys;
}

void check_all_keys_within_eps(const std::vector<li::Key>& keys, double eps) {
    auto specs = segment_stream(keys, eps);

    uint64_t total = 0;
    uint64_t expected_base = 0;
    for (const auto& s : specs) {
        EXPECT_EQ(s.base_rank, expected_base)
            << "segment base_rank not contiguous";
        EXPECT_GT(s.count, 0u) << "empty segment emitted";
        expected_base += s.count;
        total += s.count;
    }
    EXPECT_EQ(total, keys.size()) << "segments do not cover all keys";

    uint64_t gi = 0;
    for (const auto& s : specs) {
        for (uint64_t local = 0; local < s.count; ++local, ++gi) {
            const li::Key k = keys[gi];
            const double x = static_cast<double>(k - s.key_low);
            const double local_pred = s.model.alpha * x + s.model.beta;
            const double global_pred = static_cast<double>(s.base_rank) + local_pred;
            const double residual = std::fabs(global_pred - static_cast<double>(gi));
            EXPECT_LE(residual, eps + 1e-9)
                << "key " << k << " at global rank " << gi
                << " predicted " << global_pred
                << " (residual " << residual << " > eps " << eps << ")";
        }
    }
}

bool brute_coverable(const std::vector<li::Key>& keys,
                     uint64_t a, uint64_t b, double eps) {
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

uint64_t brute_segment_count(const std::vector<li::Key>& keys, double eps) {
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

void check_minimal(const std::vector<li::Key>& keys, double eps) {
    uint64_t fast = segment_stream(keys, eps).size();
    uint64_t opt  = brute_segment_count(keys, eps);
    EXPECT_EQ(fast, opt)
        << "segment count not optimal: got " << fast << ", optimal is " << opt
        << " (eps=" << eps << ")";
}
TEST(Segmentation, ExactLineIsOneSegment) {
    std::vector<li::Key> keys;
    for (uint64_t i = 0; i < 100; ++i) keys.push_back(1000 + i);
    check_all_keys_within_eps(keys, 4.0);
    EXPECT_EQ(segment_stream(keys, 4.0).size(), 1u);
}

TEST(Segmentation, CurvedDataStillWithinEps) {
    std::vector<li::Key> keys;
    uint64_t k = 0;
    for (uint64_t i = 0; i < 500; ++i) {
        k += 1 + (i / 50);
        keys.push_back(k);
    }
    check_all_keys_within_eps(keys, 8.0);
}

TEST(Segmentation, TightEpsManySegments) {
    std::vector<li::Key> keys;
    uint64_t k = 0;
    for (uint64_t i = 0; i < 300; ++i) {
        k += 1 + (i % 7);
        keys.push_back(k);
    }
    check_all_keys_within_eps(keys, 0.5);
}

TEST(Segmentation, SingleAndTwoKey) {
    check_all_keys_within_eps({42}, 4.0);
    check_all_keys_within_eps({42, 99}, 4.0);
}

TEST(Segmentation, Empty) {
    EXPECT_TRUE(segment_stream({}, 4.0).empty());
}

TEST(Segmentation, MinimalOnExactLine)  { std::vector<li::Key> k; for (uint64_t i=0;i<100;++i) k.push_back(1000+i); check_minimal(k, 4.0); }
TEST(Segmentation, MinimalOnCurved)     { std::vector<li::Key> k; uint64_t v=0; for (uint64_t i=0;i<200;++i){v+=1+(i/40);k.push_back(v);} check_minimal(k, 8.0); }
TEST(Segmentation, MinimalOnSawtooth)   { std::vector<li::Key> k; uint64_t v=0; for (uint64_t i=0;i<200;++i){v+=1+(i%13);k.push_back(v);} check_minimal(k, 2.0); }
TEST(Segmentation, MinimalTightEps)     { std::vector<li::Key> k; uint64_t v=0; for (uint64_t i=0;i<150;++i){v+=1+(i%7);k.push_back(v);} check_minimal(k, 0.5); }
TEST(Segmentation, MinimalAcrossEps) {
    std::vector<li::Key> k; uint64_t v=0;
    for (uint64_t i=0;i<300;++i){ v += 1 + ((i*2654435761u) >> 29); k.push_back(v); }
    for (double eps : {0.0, 0.5, 1.0, 2.0, 4.0}) check_minimal(k, eps);

}

TEST(Segmentation, RandomizedMinimality) {
    std::mt19937_64 rng(0xC0FFEE);

    const uint64_t sizes[]    = {1, 2, 5, 20, 80, 200};
    const uint64_t max_gaps[] = {1, 3, 10, 50, 500};
    const double   epsilons[] = {0.0, 0.5, 1.0, 2.0, 8.0};

    int trials_per_config = 50;

    for (uint64_t n : sizes) {
        for (uint64_t g : max_gaps) {
            for (double eps : epsilons) {
                for (int t = 0; t < trials_per_config; ++t) {
                    auto keys = random_keys(rng, n, g);
                    uint64_t fast = segment_stream(keys, eps).size();
                    uint64_t opt  = brute_segment_count(keys, eps);
                    ASSERT_EQ(fast, opt)
                        << "NON-OPTIMAL: n=" << n << " max_gap=" << g
                        << " eps=" << eps << " trial=" << t
                        << " -> fast=" << fast << " opt=" << opt
                        << "\nkeys: " << [&]{
                               std::string s;
                               for (auto k : keys) { s += std::to_string(k); s += ' '; }
                               return s;
                           }();
                }
            }
        }
    }
}

}