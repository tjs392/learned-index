/*
* Tree Segments
* They hold the base, size, moments, and prediction model
*/

#pragma once

#include "types.hpp"
#include "model.hpp"
#include "status.hpp"

#include <vector>
#include <cstdint>
#include <limits>
#include <cmath>

namespace li::detail {

/*
*
* OptimalPLR streaming fitter for epsilon bounded PLA
* Algorithm from "Maximum Error Bounded Piecewise Linear Representation for Online Stream Approximation"
* https://dl.acm.org/doi/10.1007/s00778-014-0355-0https://dl.acm.org/doi/pdf/10.1007/s00778-014-0355-0
* 
* The four corner rectangle cone, hull vector with start index, and slope comparison
* feasibility test refers to PGM index's optimal piecewise linear model.
*
*/

struct FittedSegment {
    Key key_low;
    uint64_t base_rank;
    uint64_t count;
    LinearModel model;
    LeastSquaresSums moments;
    Key first_key;
    Key last_key;
};

class StreamingCone {
public:
    StreamingCone(Key key_low, double epsilon) : key_low_(key_low), eps_(epsilon), n_(0) {}

    bool try_extend(Key key, uint64_t local_rank) {
        LI_ASSERT(key >= key_low_);
        const double x = static_cast<double>(key - key_low_);
        const double y = static_cast<double>(local_rank);
        const Pt p1{ x, y + eps_ };
        const Pt p2{ x, y - eps_ };

        if (n_ == 0) {
            first_key_ = key;
            last_key_ = key;
            rect_[0] = p1;
            rect_[1] = p2;
            upper_.clear();
            lower_.clear();
            upper_.push_back(p1);
            lower_.push_back(p2);
            upper_start_ = lower_start_ = 0;
            moments_.add(x, y);
            n_ = 1;
            return true;
        }

        // rank is monotone increasing :)
        if (n_ == 1) {
            rect_[2] = p2;
            rect_[3] = p1;
            upper_.push_back(p1);
            lower_.push_back(p2);
            last_key_ = key;
            moments_.add(x, y);
            n_ = 2;
            return true;
        }

        const double min_slope = edge_slope(rect_[0], rect_[2]);
        const double max_slope = edge_slope(rect_[1], rect_[3]);
        
        if (edge_slope(rect_[2], p1) < min_slope ||
            edge_slope(rect_[3], p2) > max_slope) {
            return false;
        }

        if (edge_slope(rect_[1], p1) < max_slope) {
            double best = edge_slope(lower_[lower_start_], p1);
            std::size_t best_i = lower_start_;
            for (std::size_t i = lower_start_ + 1; i < lower_.size(); ++i) {
                double v = edge_slope(lower_[i], p1);
                if (v > best) break;
                best = v;
                best_i = i;
            }

            rect_[1] = lower_[best_i];
            rect_[3] = p1;
            lower_start_ = best_i;

            std::size_t end = upper_.size();
            for (; end >= upper_start_ + 2 && cross(upper_[end - 2], upper_[end - 1], p1) <= 0.0; --end) {}
            upper_.resize(end);
            upper_.push_back(p1);
        }

        if (edge_slope(rect_[0], p2) > min_slope) {
            double best = edge_slope(upper_[upper_start_], p2);
            std::size_t best_i = upper_start_;
            for (std::size_t i = upper_start_ + 1; i < upper_.size(); ++i) {
                double v = edge_slope(upper_[i], p2);
                if (v < best) break;
                best = v;
                best_i = i;
            }
            rect_[0] = upper_[best_i];
            rect_[2] = p2;
            upper_start_ = best_i;
            std::size_t end = lower_.size();
            for (; end >= lower_start_ + 2 && cross(lower_[end - 2], lower_[end - 1], p2) >= 0.0; --end) {}
            lower_.resize(end);
            lower_.push_back(p2);
        }

        last_key_ = key;
        moments_.add(x, y);
        ++n_;
        return true;
    }

    FittedSegment finalize() {
        LI_ASSERT(n_ > 0);
        FittedSegment spec;
        spec.key_low = key_low_;
        spec.base_rank = 0;
        spec.count = n_;
        spec.first_key = first_key_;
        spec.last_key = last_key_;
        spec.moments = moments_;
        spec.model = read_off_model();
        return spec;
    }

private:
    struct Pt { 
        double x; 
        double y; 
    };

    static double edge_slope(const Pt& a, const Pt& b) {
        return (b.y - a.y) / (b.x - a.x);
    }

    static double cross(const Pt& o, const Pt& a, const Pt& b) {
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    }

    LinearModel read_off_model() const {
        if (n_ == 1) {
            return LinearModel { 0.0, 0.0 };
        }

        const double min_slope = edge_slope(rect_[0], rect_[2]);
        const double max_slope = edge_slope(rect_[1], rect_[3]);
        const double slope = 0.5 * (min_slope + max_slope);

        double ix, iy;
        // TODO: use bigints instead probably here. floats are good for now
        // but can introduce floating points error on bad data
        if (std::fabs(max_slope - min_slope) < 1e-15) {
            ix = rect_[0].x;
            iy = rect_[0].y;
        } else {
            ix = (rect_[1].y - max_slope * rect_[1].x - rect_[0].y + min_slope * rect_[0].x) / (min_slope - max_slope);
            iy = rect_[0].y + min_slope * (ix - rect_[0].x);
        }

        const double beta = iy - slope * ix;
        return LinearModel { slope, beta };
    }

    Key key_low_;
    double eps_;
    uint64_t n_;

    std::vector<Pt> upper_;
    std::vector<Pt> lower_;
    std::size_t upper_start_{0};
    std::size_t lower_start_{0};
    Pt rect_[4]{};

    Key first_key_{};
    Key last_key_{};
    LeastSquaresSums moments_{};
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
            cone = StreamingCone(keys[i], epsilon);
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