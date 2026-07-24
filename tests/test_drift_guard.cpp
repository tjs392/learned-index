// test_drift_guard.cpp - directionality, exactness on appends, and randomized
// soundness of the guard's bounds against a full rescan

#include <gtest/gtest.h>

#include "cedar/drift_guard.hpp"
#include "cedar/segmentation.hpp"

#include <vector>
#include <random>
#include <algorithm>

using namespace li;
using li::detail::DriftGuard;

namespace {

constexpr double kSlack = 1e-9;

struct RefSegment {
    LinearModel model;
    Key key_low = 0;
    std::vector<Key> keys;

    double dev_at(std::size_t rank) const {
        return line_at(model, keys[rank], key_low) - double(rank);
    }

    double true_min() const {
        double m = 1e300;
        for (std::size_t r = 0; r < keys.size(); ++r) m = std::min(m, dev_at(r));
        return m;
    }

    double true_max() const {
        double m = -1e300;
        for (std::size_t r = 0; r < keys.size(); ++r) m = std::max(m, dev_at(r));
        return m;
    }
};

void expect_sound(const DriftGuard& g, const RefSegment& s) {
    if (s.keys.empty()) return;

    EXPECT_LE(g.deviation_floor(), s.true_min() + kSlack);
    EXPECT_GE(g.deviation_ceiling(), s.true_max() - kSlack);
}

}

TEST(DriftGuard, ResetTakesExactBounds) {
    DriftGuard g;
    g.reset(-1.5, 2.5);

    EXPECT_EQ(g.deviation_floor(), -1.5);
    EXPECT_EQ(g.deviation_ceiling(), 2.5);
    EXPECT_TRUE(g.holds(2.5));
    EXPECT_FALSE(g.holds(2.0));
}

TEST(DriftGuard, HoldsBoundaryIsInclusive) {
    DriftGuard g;
    g.reset(-4.0, 4.0);

    EXPECT_TRUE(g.holds(4.0));

    g.reset(-4.0 - 1e-12, 4.0);

    EXPECT_FALSE(g.holds(4.0));
}

TEST(DriftGuard, AppendsAreExact) {
    RefSegment s;
    s.model = LinearModel{ 1.0 / 10.0, 0.0 };
    s.key_low = 100;

    DriftGuard g;
    g.reset(0.0, 0.0);
    s.keys.push_back(100);

    std::mt19937_64 rng(1);
    Key k = 100;

    for (int i = 0; i < 400; ++i) {
        k += 5 + rng() % 12;
        const std::size_t new_rank = s.keys.size();
        const double arriving = line_at(s.model, k, s.key_low) - double(new_rank);

        (void)g.try_absorb_insert(arriving, 1e9, false);
        s.keys.push_back(k);

        // decay = 0 on appends, so the bounds stay exact, not just sound
        EXPECT_NEAR(g.deviation_floor(), s.true_min(), 1e-9);
        EXPECT_NEAR(g.deviation_ceiling(), s.true_max(), 1e-9);
    }
}

TEST(DriftGuard, MiddleInsertDecaysFloorOnly) {
    DriftGuard g;
    g.reset(-1.0, 1.0);

    g.try_absorb_insert(0.25, 1e9, true);

    EXPECT_EQ(g.deviation_floor(), -2.0);
    EXPECT_EQ(g.deviation_ceiling(), 1.0);

    g.try_absorb_insert(3.0, 1e9, true);

    EXPECT_EQ(g.deviation_floor(), -3.0);
    EXPECT_EQ(g.deviation_ceiling(), 3.0);
}

TEST(DriftGuard, EraseDecaysCeilingOnly) {
    DriftGuard g;
    g.reset(-1.0, 1.0);

    g.try_absorb_erase(1e9, true);

    EXPECT_EQ(g.deviation_floor(), -1.0);
    EXPECT_EQ(g.deviation_ceiling(), 2.0);

    g.try_absorb_erase(1e9, false);

    EXPECT_EQ(g.deviation_floor(), -1.0);
    EXPECT_EQ(g.deviation_ceiling(), 2.0);
}

TEST(DriftGuard, RandomizedSoundnessVsRescan) {
    std::mt19937_64 rng(20260724);

    for (int round = 0; round < 200; ++round) {
        RefSegment s;
        s.key_low = 1000;
        s.model = LinearModel{ 1.0 / double(1 + rng() % 30), double(rng() % 5) };

        Key k = s.key_low;
        const std::size_t seed_n = 2 + rng() % 60;

        for (std::size_t i = 0; i < seed_n; ++i) {
            s.keys.push_back(k);
            k += 1 + rng() % 40;
        }

        DriftGuard g;
        g.reset(s.true_min(), s.true_max());
        expect_sound(g, s);

        for (int op = 0; op < 120; ++op) {
            const unsigned dice = unsigned(rng() % 100);

            if (dice < 55) {
                const std::size_t pos = rng() % (s.keys.size() + 1);
                const Key lo = (pos == 0) ? s.key_low : s.keys[pos - 1];
                const Key hi = (pos == s.keys.size()) ? lo + 100 : s.keys[pos];

                if (hi <= lo + 1) continue;

                const Key nk = lo + 1 + rng() % (hi - lo - 1);
                const bool shifted = pos + 1 < s.keys.size() + 1;
                const double arriving = line_at(s.model, nk, s.key_low) - double(pos);

                g.try_absorb_insert(arriving, 1e9, shifted);
                s.keys.insert(s.keys.begin() + std::ptrdiff_t(pos), nk);
            } else if (s.keys.size() > 1) {
                const std::size_t pos = rng() % s.keys.size();
                const bool shifted = pos + 1 < s.keys.size();

                g.try_absorb_erase(1e9, shifted);
                s.keys.erase(s.keys.begin() + std::ptrdiff_t(pos));
            }

            expect_sound(g, s);

            if (::testing::Test::HasFailure()) return;
        }
    }
}
