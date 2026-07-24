// test_segmentation.cpp - the cone against brute force: accepted prefixes must
// carry an eps-valid model, rejects must be real splits, reset must equal a
// fresh cone, and the greedy cover must be maximal

#include <gtest/gtest.h>

#include "cedar/segmentation.hpp"

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

using namespace li;
using li::detail::StreamingCone;
using li::detail::FittedSegment;
using li::detail::LineCoverStatus;

namespace {

constexpr double kSlack = 1e-6;

double max_abs_dev(const LinearModel& m, Key key_low,
                   const std::vector<Key>& keys, std::size_t begin, std::size_t end) {
    double worst = 0.0;

    for (std::size_t i = begin; i < end; ++i) {
        const double dev = line_at(m, keys[i], key_low) - double(i - begin);
        worst = std::max(worst, std::fabs(dev));
    }

    return worst;
}

std::vector<Key> random_stream(std::mt19937_64& rng, std::size_t n, int shape) {
    std::vector<Key> keys;
    Key k = 1 + rng() % 1000;

    for (std::size_t i = 0; i < n; ++i) {
        keys.push_back(k);

        if (shape == 0) k += 1 + rng() % 40;
        else if (shape == 1) k += (i % 40 < 36) ? 1 + rng() % 3 : 300 + rng() % 3000;
        else k += (rng() % 12 == 0) ? 1 + rng() % 100000 : 1 + rng() % 10;
    }

    return keys;
}

}

TEST(Segmentation, PredictClampsAtZero) {
    const LinearModel m{ 1.0, -5.0 };

    EXPECT_EQ(predict(m, 100, 100), 0u);
    EXPECT_EQ(predict(m, 110, 100), 5u);
    EXPECT_EQ(line_at(m, 110, 100), 5.0);
}

TEST(Segmentation, AcceptedPrefixAlwaysHasEpsValidModel) {
    std::mt19937_64 rng(21);

    for (int trial = 0; trial < 60; ++trial) {
        const double eps = (trial % 3 == 0) ? 0.5 : double(1 + rng() % 64);
        const std::vector<Key> keys = random_stream(rng, 200, trial % 3);

        StreamingCone cone(keys[0], eps);
        std::size_t accepted = 0;

        for (std::size_t i = 0; i < keys.size(); ++i) {
            if (!cone.try_extend(keys[i], accepted)) break;
            ++accepted;

            if (accepted % 17 == 0) {
                const FittedSegment spec = cone.finalize();

                EXPECT_LE(max_abs_dev(spec.model, keys[0], keys, 0, accepted), eps + kSlack);
            }
        }

        ASSERT_GT(accepted, 0u);

        const FittedSegment spec = cone.finalize();

        EXPECT_EQ(spec.count, accepted);
        EXPECT_EQ(spec.key_low, keys[0]);
        EXPECT_EQ(spec.first_key, keys[0]);
        EXPECT_EQ(spec.last_key, keys[accepted - 1]);
        EXPECT_LE(max_abs_dev(spec.model, keys[0], keys, 0, accepted), eps + kSlack);

        if (::testing::Test::HasFailure()) return;
    }
}

TEST(Segmentation, RejectMeansNoSingleLineExists) {
    std::mt19937_64 rng(22);

    for (int trial = 0; trial < 60; ++trial) {
        const double eps = double(1 + rng() % 8);
        const std::vector<Key> keys = random_stream(rng, 300, trial % 3);

        StreamingCone cone(keys[0], eps);
        std::size_t accepted = 0;

        while (accepted < keys.size() && cone.try_extend(keys[accepted], accepted)) ++accepted;

        if (accepted == keys.size()) continue;

        std::span<const Key> broken(keys.data(), accepted + 1);

        EXPECT_EQ(detail::minimal_line_cover(broken, eps).status, LineCoverStatus::SPLIT);

        std::span<const Key> fine(keys.data(), accepted);

        EXPECT_EQ(detail::minimal_line_cover(fine, eps).status, LineCoverStatus::COVERABLE);

        if (::testing::Test::HasFailure()) return;
    }
}

TEST(Segmentation, ResetIsEquivalentToFreshCone) {
    std::mt19937_64 rng(23);

    for (int trial = 0; trial < 40; ++trial) {
        const double eps = (trial % 2 == 0) ? 0.5 : 4.0;
        const std::vector<Key> keys = random_stream(rng, 400, trial % 3);

        StreamingCone reused(keys[0], eps);
        std::size_t segment_start = 0;

        for (std::size_t i = 0; i < keys.size(); ++i) {
            const std::uint64_t local = i - segment_start;

            if (reused.try_extend(keys[i], local)) continue;

            StreamingCone fresh(keys[segment_start], eps);

            for (std::size_t j = segment_start; j < i; ++j) {
                ASSERT_TRUE(fresh.try_extend(keys[j], j - segment_start));
            }

            EXPECT_FALSE(fresh.try_extend(keys[i], local));

            const LinearModel a = reused.finalize().model;
            const LinearModel b = fresh.finalize().model;

            EXPECT_EQ(a.alpha, b.alpha);
            EXPECT_EQ(a.beta, b.beta);

            segment_start = i;
            reused.reset(keys[i]);
            ASSERT_TRUE(reused.try_extend(keys[i], 0));
        }

        if (::testing::Test::HasFailure()) return;
    }
}

TEST(Segmentation, SegmentStreamPartitionsAndIsGreedyMaximal) {
    std::mt19937_64 rng(24);

    for (int trial = 0; trial < 40; ++trial) {
        const double eps = double(1 + rng() % 16);
        const std::vector<Key> keys = random_stream(rng, 500, trial % 3);

        const std::vector<FittedSegment> specs = detail::segment_stream(keys, eps);

        ASSERT_FALSE(specs.empty());

        std::size_t covered = 0;

        for (std::size_t s = 0; s < specs.size(); ++s) {
            const FittedSegment& sp = specs[s];

            EXPECT_EQ(sp.base_rank, covered);
            ASSERT_GT(sp.count, 0u);
            EXPECT_EQ(sp.key_low, keys[covered]);
            EXPECT_EQ(sp.first_key, keys[covered]);
            EXPECT_EQ(sp.last_key, keys[covered + sp.count - 1]);
            EXPECT_LE(max_abs_dev(sp.model, sp.key_low, keys, covered, covered + sp.count),
                      eps + kSlack);

            if (s + 1 < specs.size()) {
                StreamingCone cone(sp.key_low, eps);

                for (std::size_t j = 0; j < sp.count; ++j) {
                    ASSERT_TRUE(cone.try_extend(keys[covered + j], j));
                }

                EXPECT_FALSE(cone.try_extend(keys[covered + sp.count], sp.count));
            }

            covered += sp.count;
        }

        EXPECT_EQ(covered, keys.size());

        if (::testing::Test::HasFailure()) return;
    }
}

TEST(Segmentation, MinimalLineCoverPiecesPartitionAndFit) {
    std::mt19937_64 rng(25);

    for (int trial = 0; trial < 40; ++trial) {
        const double eps = double(1 + rng() % 16);
        const std::vector<Key> keys = random_stream(rng, 400, trial % 3);

        const auto result = detail::minimal_line_cover(keys, eps);

        std::size_t covered = 0;

        for (const auto& piece : result.pieces) {
            EXPECT_EQ(piece.begin, covered);
            ASSERT_GT(piece.end, piece.begin);
            EXPECT_LE(max_abs_dev(piece.model, keys[piece.begin], keys, piece.begin, piece.end),
                      eps + kSlack);
            covered = piece.end;
        }

        EXPECT_EQ(covered, keys.size());

        if (result.status == LineCoverStatus::COVERABLE) {
            EXPECT_EQ(result.pieces.size(), 1u);
            EXPECT_EQ(result.model.alpha, result.pieces.front().model.alpha);
            EXPECT_EQ(result.model.beta, result.pieces.front().model.beta);
        } else {
            EXPECT_GT(result.pieces.size(), 1u);
        }

        if (::testing::Test::HasFailure()) return;
    }
}

TEST(Segmentation, ExactAffineStreamNeedsZeroEps) {
    std::vector<Key> keys;

    for (Key i = 0; i < 300; ++i) keys.push_back(1000 + 7 * i);

    EXPECT_EQ(detail::minimal_line_cover(keys, 0.0).status, LineCoverStatus::COVERABLE);

    keys[150] += 3;
    std::sort(keys.begin(), keys.end());

    EXPECT_EQ(detail::minimal_line_cover(keys, 0.0).status, LineCoverStatus::SPLIT);
}

TEST(Segmentation, FractionalEpsIsExact) {
    std::mt19937_64 rng(26);

    for (int trial = 0; trial < 30; ++trial) {
        const std::vector<Key> keys = random_stream(rng, 300, trial % 3);
        const auto result = detail::minimal_line_cover(keys, 0.5);

        for (const auto& piece : result.pieces) {
            EXPECT_LE(max_abs_dev(piece.model, keys[piece.begin], keys, piece.begin, piece.end),
                      0.5 + kSlack);
        }

        if (::testing::Test::HasFailure()) return;
    }
}
