#include <gtest/gtest.h>
#include "li/minimal_line_cover.hpp"
#include "li/segmentation.hpp"
#include "li/model.hpp"

#include <vector>
#include <span>
#include <random>
#include <cmath>
#include <cstdint>

namespace {

using li::detail::minimal_line_cover;
using li::detail::LineCoverStatus;
using li::detail::LineCoverResult;
using li::detail::LineCoverPiece;
using li::detail::segment_stream;
using li::Key;

constexpr double kEpsSlack = 1e-9;

void expect_piece_in_band(std::span<const Key> keys, const LineCoverPiece& piece, double eps) {
    const Key key_low = keys[piece.begin];
    for (std::size_t i = piece.begin; i < piece.end; ++i) {
        const double dev = li::line_at(piece.model, keys[i], key_low) - double(i - piece.begin);
        EXPECT_LE(std::fabs(dev), eps + kEpsSlack)
            << "piece [" << piece.begin << "," << piece.end << ") key " << keys[i]
            << " local_rank " << (i - piece.begin) << " dev " << dev << " eps " << eps;
    }
}

void expect_tiles(const LineCoverResult& r, std::size_t n) {
    ASSERT_FALSE(r.pieces.empty());
    EXPECT_EQ(r.pieces.front().begin, 0u);
    EXPECT_EQ(r.pieces.back().end, n);
    for (std::size_t i = 1; i < r.pieces.size(); ++i)
        EXPECT_EQ(r.pieces[i - 1].end, r.pieces[i].begin) << "gap/overlap at piece " << i;
    for (const auto& p : r.pieces) EXPECT_LT(p.begin, p.end) << "empty piece range";
}

void expect_matches_segment_stream(const std::vector<Key>& keys, double eps) {
    LineCoverResult r = minimal_line_cover(keys, eps);
    auto specs = segment_stream(keys, eps);

    ASSERT_EQ(r.pieces.size(), specs.size()) << "piece count disagrees with segment_stream";
    for (std::size_t i = 0; i < specs.size(); ++i) {
        EXPECT_EQ(r.pieces[i].begin, specs[i].base_rank) << "piece " << i << " start";
        EXPECT_EQ(r.pieces[i].end, specs[i].base_rank + specs[i].count) << "piece " << i << " end";
        EXPECT_EQ(r.pieces[i].model.alpha, specs[i].model.alpha) << "piece " << i << " alpha";
        EXPECT_EQ(r.pieces[i].model.beta, specs[i].model.beta) << "piece " << i << " beta";
    }

    EXPECT_EQ(r.status == LineCoverStatus::COVERABLE, r.pieces.size() == 1u);
    expect_tiles(r, keys.size());
    for (const auto& p : r.pieces) expect_piece_in_band(keys, p, eps);
}

TEST(MinimalLineCover, SingleKeyIsCoverable) {
    std::vector<Key> keys = {42};
    LineCoverResult r = minimal_line_cover(keys, 1.0);
    EXPECT_EQ(r.status, LineCoverStatus::COVERABLE);
    ASSERT_EQ(r.pieces.size(), 1u);
    expect_matches_segment_stream(keys, 1.0);
}

TEST(MinimalLineCover, TwoKeysAlwaysOneLine) {
    std::vector<Key> keys = {10, 1000000};
    LineCoverResult r = minimal_line_cover(keys, 0.0);
    EXPECT_EQ(r.status, LineCoverStatus::COVERABLE) << "any two points define a line";
    expect_matches_segment_stream(keys, 0.0);
}

TEST(MinimalLineCover, ExactLineIsOnePiece) {
    std::vector<Key> keys;
    for (Key k = 0; k < 200; ++k) keys.push_back(5 + 3 * k);
    LineCoverResult r = minimal_line_cover(keys, 0.0);
    EXPECT_EQ(r.status, LineCoverStatus::COVERABLE);
    EXPECT_EQ(r.pieces.size(), 1u);
    expect_matches_segment_stream(keys, 0.0);
}

TEST(MinimalLineCover, OffTrendForcesSplit) {
    std::vector<Key> keys = {0, 10, 20, 30, 40, 41};
    LineCoverResult r = minimal_line_cover(keys, 0.0);
    EXPECT_EQ(r.status, LineCoverStatus::SPLIT);
    expect_matches_segment_stream(keys, 0.0);
}

TEST(MinimalLineCover, StaircaseForcesManyPieces) {
    std::vector<Key> keys;
    Key v = 0;
    for (int block = 0; block < 5; ++block) {
        for (int j = 0; j < 4; ++j) keys.push_back(v++);
        v += 1000;
    }
    expect_matches_segment_stream(keys, 0.0);
}

TEST(MinimalLineCover, RandomDifferentialVsSegmentStream) {
    for (double eps : {0.0, 0.5, 1.0, 2.0, 8.0}) {
        std::mt19937_64 rng(2024 + uint64_t(eps * 10));
        for (int trial = 0; trial < 500; ++trial) {
            std::size_t n = 1 + (rng() % 300);
            std::vector<Key> keys;
            Key v = rng() % 100;
            for (std::size_t i = 0; i < n; ++i) { v += 1 + (rng() % 30); keys.push_back(v); }
            expect_matches_segment_stream(keys, eps);
        }
    }
}

}