#include <gtest/gtest.h>
#include "li/hull.hpp"
#include "li/segmentation.hpp"
#include "li/model.hpp"

#include <vector>
#include <random>
#include <cmath>
#include <cstdint>

namespace {

using li::detail::StaticHull;
using li::detail::segment_stream;
using li::Key;

constexpr double kSlack = 1e-9;

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

std::vector<Key> random_keys(std::mt19937_64& rng, uint64_t n, uint64_t max_gap) {
    std::vector<Key> keys;
    keys.reserve(n);
    uint64_t k = 1 + (rng() % max_gap);
    for (uint64_t i = 0; i < n; ++i) { keys.push_back(k); k += 1 + (rng() % max_gap); }
    return keys;
}

void expect_model_in_band(const std::vector<Key>& keys, double eps) {
    StaticHull h = StaticHull::build(keys, eps);
    if (!h.is_coverable()) return;
    auto m = h.read_off_model();
    const Key kl = h.key_low();
    for (uint64_t i = 0; i < keys.size(); ++i) {
        double pred = m.alpha * static_cast<double>(keys[i] - kl) + m.beta;
        EXPECT_LE(std::fabs(pred - static_cast<double>(i)), eps + kSlack)
            << "model out of band at i=" << i << " key=" << keys[i] << " eps=" << eps;
    }
}

TEST(HullFeasibility, AgreesWithOracleAndSegmentStream) {
    std::mt19937_64 rng(1);
    const uint64_t sizes[] = {1, 2, 3, 5, 20, 80, 200};
    const uint64_t gaps[]  = {1, 2, 5, 40, 400};
    const double   epss[]  = {0.0, 0.5, 1.0, 2.0, 8.0};

    for (uint64_t n : sizes) {
        for (uint64_t g : gaps) {
            for (double eps : epss) {
                for (int t = 0; t < 40; ++t) {
                    auto keys = random_keys(rng, n, g);
                    const bool hull_cov = StaticHull::build(keys, eps).is_coverable();
                    const bool stream_cov = segment_stream(keys, eps).size() == 1;
                    EXPECT_EQ(hull_cov, stream_cov)
                        << "hull vs segment_stream disagree: n=" << n << " g=" << g << " eps=" << eps;
                    if (n >= 1) {
                        const bool brute_cov = brute_coverable(keys, 0, keys.size() - 1, eps);
                        EXPECT_EQ(hull_cov, brute_cov)
                            << "hull vs brute disagree: n=" << n << " g=" << g << " eps=" << eps;
                    }
                }
            }
        }
    }
}

TEST(HullFeasibility, ModelInBand) {
    std::mt19937_64 rng(7);
    for (double eps : {0.0, 0.5, 1.0, 2.0, 8.0}) {
        for (int t = 0; t < 200; ++t) {
            expect_model_in_band(random_keys(rng, 1 + (rng() % 120), 1 + (rng() % 50)), eps);
        }
    }
}

TEST(HullFeasibility, ForcedCutIsMaximalPrefix) {
    std::mt19937_64 rng(99);
    for (double eps : {0.0, 0.5, 1.0}) {
        for (int t = 0; t < 500; ++t) {
            auto keys = random_keys(rng, 3 + (rng() % 120), 1 + (rng() % 60));
            StaticHull h = StaticHull::build(keys, eps);
            if (h.is_coverable()) continue;
            const std::size_t cut = h.forced_cut_index();
            ASSERT_GT(cut, 0u);
            ASSERT_LT(cut, keys.size());
            EXPECT_TRUE(brute_coverable(keys, 0, cut - 1, eps)) << "prefix [0,cut) not coverable, cut=" << cut;
            EXPECT_FALSE(brute_coverable(keys, 0, cut, eps)) << "prefix [0,cut] coverable, cut=" << cut;
        }
    }
}

TEST(HullFeasibility, HullsAreConvexChains) {
    std::mt19937_64 rng(123);
    for (int t = 0; t < 300; ++t) {
        auto keys = random_keys(rng, 2 + (rng() % 100), 1 + (rng() % 40));
        StaticHull h = StaticHull::build(keys, 1.0);
        const auto& lo = h.hull_lo();
        const auto& hi = h.hull_hi();
        for (std::size_t i = 0; i + 1 < lo.size(); ++i) EXPECT_LT(lo[i].x, lo[i + 1].x);
        for (std::size_t i = 0; i + 1 < hi.size(); ++i) EXPECT_LT(hi[i].x, hi[i + 1].x);
        auto crossp = [](const StaticHull::Pt& o, const StaticHull::Pt& a, const StaticHull::Pt& b) {
            return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
        };
        for (std::size_t i = 0; i + 2 < lo.size(); ++i)
            EXPECT_LT(crossp(lo[i], lo[i + 1], lo[i + 2]), 1e-9) << "hull_lo not upper-convex at " << i;
        for (std::size_t i = 0; i + 2 < hi.size(); ++i)
            EXPECT_GT(crossp(hi[i], hi[i + 1], hi[i + 2]), -1e-9) << "hull_hi not lower-convex at " << i;
    }
}

}