/*
* segmentation.hpp - The model, cone, and cover
*
* LinearModel + line_at/predict/residual: the line and its evaluations
* StreamingCone + segment_stream: exact arithmetic OptimalPLR cone, certifies
* eps coverability and streams maximal segments
* minimal_line_cover: coverable with one model, or the forced cut pieces
* only called on a break, irreducibility bounds a split to 1 <= n <= 3 pieces
*
*
* inspired by:
* OptimalPLR:https://dl.acm.org/doi/10.1007/s00778-014-0355-0
* PGM index
*
*/

#pragma once

#include "status.hpp"

#include <vector>
#include <cstdint>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <span>
#include <cstddef>

namespace li {

// zeroing is load bearing because split branch leaves model unassigned
struct LinearModel {
    double alpha = 0.0;
    double beta = 0.0;
};


[[nodiscard]] inline Rank predict(LinearModel m, Key k, Key key_low) {
    CEDAR_ASSERT(k >= key_low);

    double x = static_cast<double>(k - key_low);
    double p = m.alpha * x + m.beta;

    if (p < 0.0) p = 0.0;

    return static_cast<Rank>(p);
}


[[nodiscard]] inline double line_at(LinearModel m, Key key, Key key_low) {
    CEDAR_ASSERT(key >= key_low);

    return m.alpha * static_cast<double>(key - key_low) + m.beta;
}

}


namespace li::detail {

/*
*
* **IMPORTANT**
* doubles are not enough here: products
* reach ~4e17 where double carries absolute error ~40, so the sign of a
* predicate can be wrong at key spans near 2^60 (i learned this the hard way :( )
*/

struct FittedSegment {
    Key key_low;
    uint64_t base_rank;
    uint64_t count;
    LinearModel model;
    Key first_key;
    Key last_key;
};

class StreamingCone {
public:

    StreamingCone(Key key_low, double epsilon) : key_low_(key_low), count_(0) {
        CEDAR_ASSERT(epsilon >= 0.0);

        upper_hull_.reserve(kHullReserve);
        lower_hull_.reserve(kHullReserve);

        int bits = 0;
        double scaled = epsilon;

        while (scaled != std::floor(scaled) && bits < kMaxScaleBits) {
            scaled *= 2.0;
            ++bits;
        }

        rank_scale_ = int64_t(1) << bits;
        epsilon_scaled_ = static_cast<int64_t>(std::floor(scaled));
    }

    void reset(Key key_low) {
        key_low_ = key_low;
        count_ = 0;
    }
    

    bool try_extend(Key key, uint64_t local_rank) {
        CEDAR_ASSERT(key >= key_low_);

        const uint64_t x = key - key_low_;
        const int64_t y = static_cast<int64_t>(local_rank) * rank_scale_;
        const Point upper_point{ x, y + epsilon_scaled_ };
        const Point lower_point{ x, y - epsilon_scaled_ };

        if (count_ == 0) {
            first_key_ = key;
            last_key_ = key;
            shallowest_from_ = upper_point;
            steepest_from_ = lower_point;
            upper_hull_.clear();
            lower_hull_.clear();
            upper_hull_.push_back(upper_point);
            lower_hull_.push_back(lower_point);
            upper_live_from_ = lower_live_from_ = 0;
            count_ = 1;

            return true;
        }

        // rank is monotone increasing :)
        if (count_ == 1) {
            shallowest_to_ = lower_point;
            steepest_to_ = upper_point;
            upper_hull_.push_back(upper_point);
            lower_hull_.push_back(lower_point);
            last_key_ = key;
            count_ = 2;

            return true;
        }

        const bool below_shallowest = compare_slopes(shallowest_to_, upper_point, shallowest_from_, shallowest_to_) < 0;
        const bool above_steepest = compare_slopes(steepest_to_, lower_point, steepest_from_, steepest_to_) > 0;

        if (below_shallowest || above_steepest) {
            return false;
        }

        const bool upper_point_tightens_steepest = compare_slopes(steepest_from_, upper_point, steepest_from_, steepest_to_) < 0;

        if (upper_point_tightens_steepest) {
            std::size_t tangent = lower_live_from_;

            for (std::size_t i = lower_live_from_ + 1; i < lower_hull_.size(); ++i) {
                if (compare_slopes(lower_hull_[i], upper_point, lower_hull_[tangent], upper_point) > 0) break;
                tangent = i;
            }

            steepest_from_ = lower_hull_[tangent];
            steepest_to_ = upper_point;
            lower_live_from_ = tangent;

            std::size_t keep = upper_hull_.size();

            for (; keep >= upper_live_from_ + 2 &&
                   turn_direction(upper_hull_[keep - 2], upper_hull_[keep - 1], upper_point) <= 0; --keep) {}

            upper_hull_.resize(keep);
            upper_hull_.push_back(upper_point);
        }
        

        const bool lower_point_tightens_shallowest =
            compare_slopes(shallowest_from_, lower_point, shallowest_from_, shallowest_to_) > 0;

        if (lower_point_tightens_shallowest) {
            std::size_t tangent = upper_live_from_;

            for (std::size_t i = upper_live_from_ + 1; i < upper_hull_.size(); ++i) {
                if (compare_slopes(upper_hull_[i], lower_point, upper_hull_[tangent], lower_point) < 0) break;
                tangent = i;
            }

            shallowest_from_ = upper_hull_[tangent];
            shallowest_to_ = lower_point;
            upper_live_from_ = tangent;

            std::size_t keep = lower_hull_.size();

            for (; keep >= lower_live_from_ + 2 &&
                   turn_direction(lower_hull_[keep - 2], lower_hull_[keep - 1], lower_point) >= 0; --keep) {}

            lower_hull_.resize(keep);
            lower_hull_.push_back(lower_point);
        }

        last_key_ = key;
        ++count_;

        return true;
    }

    // at a fixed slope, how much room the intercept still has inside the eps band
    // margin = half of this: rank shifts absorbed before a key leaves the band
    double chord_at(double slope_scaled) const {
        double hi = std::numeric_limits<double>::infinity();
        double lo = -std::numeric_limits<double>::infinity();

        for (const Point& p : upper_hull_) {
            hi = std::min(hi, static_cast<double>(p.y) - slope_scaled * static_cast<double>(p.x));
        }

        for (const Point& p : lower_hull_) {
            lo = std::max(lo, static_cast<double>(p.y) - slope_scaled * static_cast<double>(p.x));
        }

        return hi - lo;
    }

    double current_margin() const {
        if (count_ < 2) return 0.0;

        const double s = 0.5 * (slope_between(shallowest_from_, shallowest_to_) +
                                slope_between(steepest_from_, steepest_to_));

        return 0.5 * chord_at(s) / double(rank_scale_);
    }

    uint64_t count() const { return count_; }

    FittedSegment finalize() {
        CEDAR_ASSERT(count_ > 0);

        FittedSegment spec;
        spec.key_low = key_low_;
        spec.base_rank = 0;
        spec.count = count_;
        spec.first_key = first_key_;
        spec.last_key = last_key_;
        spec.model = read_off_model();

        return spec;
    }


private:

    __extension__ using i128 = __int128;

    // x = key - key_low fits uint64
    // y = rank +/- eps fits int64
    struct Point {
        uint64_t x;
        int64_t y;
    };

    static int sign_of(i128 value) { 
        return (value > 0) - (value < 0); 
    }

    static i128 exact_product(uint64_t run, int64_t rise) {
        return static_cast<i128>(run) * static_cast<i128>(rise);
    }

    // positive when origin -> first -> second turns counter clockwise
    static int turn_direction(const Point& origin, const Point& first, const Point& second) {

        return sign_of(exact_product(first.x - origin.x, second.y - origin.y) -
                       exact_product(second.x - origin.x, first.y - origin.y)
        );
    }

    // sign of slope(first) - slope(second)
    static int compare_slopes(const Point& first_from, const Point& first_to,
                              const Point& second_from, const Point& second_to) {

        const int64_t first_rise = first_to.y - first_from.y;
        const uint64_t first_run = first_to.x - first_from.x;
        const int64_t second_rise = second_to.y - second_from.y;
        const uint64_t second_run = second_to.x - second_from.x;

        return sign_of(exact_product(second_run, first_rise) - exact_product(first_run, second_rise));
    }

    static double slope_between(const Point& from, const Point& to) {
        return static_cast<double>(static_cast<long double>(to.y - from.y) /
                                   static_cast<long double>(to.x - from.x));
    }

    LinearModel read_off_model() const {
        if (count_ == 1) {
            return LinearModel { 0.0, 0.0 };
        }

        // everything here is in scaled rank space, divided back at the end
        const double slope = 0.5 * (slope_between(shallowest_from_, shallowest_to_) +
                                    slope_between(steepest_from_, steepest_to_));

        // beta is bounded by every hull point at this slope
        double beta_ceiling = std::numeric_limits<double>::infinity();
        double beta_floor = -std::numeric_limits<double>::infinity();

        for (const Point& p : upper_hull_) {
            beta_ceiling = std::min(beta_ceiling, static_cast<double>(p.y) - slope * static_cast<double>(p.x));
        }

        for (const Point& p : lower_hull_) {
            beta_floor = std::max(beta_floor, static_cast<double>(p.y) - slope * static_cast<double>(p.x));
        }

        // **IMPORTANT**
        // greedy maximal segments run until the feasible window collapses, so one
        // ulp of slope rounding can invert the beta interval. the slack absorbs
        // exactly that class, a truly infeasible read off still trips the assert
        const double feasibility_slack = 1e-9 * (1.0 + std::max(std::abs(beta_floor), std::abs(beta_ceiling)));

        CEDAR_ASSERT(beta_floor <= beta_ceiling + feasibility_slack);

        const double beta = 0.5 * (beta_floor + beta_ceiling);

        return LinearModel { slope / double(rank_scale_), beta / double(rank_scale_) };
    }

    static constexpr int kMaxScaleBits = 20;

    Key key_low_;
    int64_t rank_scale_{ 1 };
    int64_t epsilon_scaled_{ 0 };
    uint64_t count_;

    static constexpr std::size_t kHullReserve = 64;

    std::vector<Point> upper_hull_;
    std::vector<Point> lower_hull_;
    std::size_t upper_live_from_{0};
    std::size_t lower_live_from_{0};

    // the four cone corners:
    // the shallowest feasible line runs upper point -> later lower point, 
    // the steepest lower -> later upper
    // together they bound every slope that still eps covers the keys seen so far
    Point shallowest_from_{};
    Point shallowest_to_{};
    Point steepest_from_{};
    Point steepest_to_{};

    Key first_key_{};
    Key last_key_{};
};

inline std::vector<FittedSegment> segment_stream(const std::vector<Key>& keys, double epsilon) {
    std::vector<FittedSegment> specs;

    if (keys.empty()) return specs;

    uint64_t base = 0;
    StreamingCone cone(keys[0], epsilon);
    cone.try_extend(keys[0], 0);

    uint64_t local = 1;

    for (uint64_t i = 1; i < keys.size(); ++i) {
        if (!cone.try_extend(keys[i], local)) {
            FittedSegment spec = cone.finalize();
            spec.base_rank = base;
            specs.push_back(spec);

            base = i;
            cone.reset(keys[i]);
            cone.try_extend(keys[i], 0);
            local = 1;
        } else {
            ++local;
        }
    }

    FittedSegment spec = cone.finalize();
    spec.base_rank = base;
    specs.push_back(spec);

    return specs;
}

}


namespace li::detail {

enum class LineCoverStatus { 
    COVERABLE, 
    SPLIT 
};

struct LineCoverPiece {
    std::size_t begin;
    std::size_t end;
    LinearModel model;
};

struct LineCoverResult {
    LineCoverStatus status;
    LinearModel model;
    std::vector<LineCoverPiece> pieces;
};

// TODO(perf):
// if we really need some performance, pieces is a heap allocated vector that gets pushed
// a bunch here.
// can use a smallvec and allocate 4 pieces. SmallVector<LineCoverPiece, 4>
// but this is constant, so measure before and after if i wnt to do tis
inline LineCoverResult minimal_line_cover(std::span<const Key> sorted_keys, double epsilon) {
    CEDAR_ASSERT(!sorted_keys.empty());

    LineCoverResult result;
    std::size_t piece_start = 0;
    StreamingCone cone(sorted_keys[0], epsilon);

    for (std::size_t i = 0; i < sorted_keys.size(); ++i) {
        std::size_t local_rank = i - piece_start;

        if (!cone.try_extend(sorted_keys[i], local_rank)) {
            result.pieces.push_back(LineCoverPiece{ piece_start, i, cone.finalize().model });

            piece_start = i;
            cone.reset(sorted_keys[i]);
            cone.try_extend(sorted_keys[i], 0);
        }
    }

    result.pieces.push_back(LineCoverPiece{ piece_start, sorted_keys.size(), cone.finalize().model });

    if (result.pieces.size() == 1) {
        result.status = LineCoverStatus::COVERABLE;
        result.model = result.pieces.front().model;
    } else {
        result.status = LineCoverStatus::SPLIT;
    }

    return result;
}

}