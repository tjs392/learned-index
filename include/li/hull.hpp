#pragma once

#include "types.hpp"
#include "model.hpp"
#include "status.hpp"
#include "segmentation.hpp"

#include <vector>
#include <span>
#include <cstddef>
#include <cstdint>

namespace li::detail {

// static hull is a geometry validator + the builder
// its chains (hull_lo/hull_hi) + moments are the rep the dynamic tree grows
// build_balanced() calls hull consturction logic per rebuilt subtree
// Pt right now is temporary
// TODO: ^^^

// DEPRACATED
// not used anymore, this is algorithmically optimal,
// but horribly cache unfriendly - so moving on from hull trees and tatic hull
class StaticHull {
public:
    struct Pt { double x; double y; };

    static StaticHull build(std::span<const Key> keys, double epsilon, double merge_slack = 0.0) {
        StaticHull h;
        h.count_ = keys.size();
        if (keys.empty()) { h
            .coverable_ = false; 
            h.forced_cut_ = 0; 
            return h; 
        }

        h.key_low_ = keys[0];
        h.eps_ = epsilon * (1.0 - merge_slack);

        for (std::size_t i = 0; i < keys.size(); ++i) {
            const double x = static_cast<double>(keys[i] - h.key_low_);
            const double y = static_cast<double>(i);
            
            h.moments_.add(x, y);
            
            h.push_hull_hi(Pt{ x, y + h.eps_ });
            h.push_hull_lo(Pt{ x, y - h.eps_ });
        }

        // TODO: this is just scaffolding
        // the feasibility logic is just borrow from streaming cone right now for correctness purposes
        // we will be deriving feasibility directly from the maintained hull using tangent query
        // this is to be deleted
        StreamingCone cone(keys[0], h.eps_);
        h.coverable_ = true;
        h.forced_cut_ = keys.size();
        for (std::size_t i = 0; i < keys.size(); ++i) {
            if (!cone.try_extend(keys[i], i)) {
                h.coverable_ = false;
                h.forced_cut_ = i;
                break;
            }
        }
        if (h.coverable_) h.model_ = cone.finalize().model;
        return h;
    }

    bool is_coverable() const { return coverable_; }
    
    std::size_t forced_cut_index() const { return forced_cut_; }
    
    std::size_t count() const { return count_; }
    
    Key key_low() const { return key_low_; }
    
    double eps() const { return eps_; }

    LinearModel read_off_model() const {
        LI_ASSERT(coverable_);
        return model_;
    }

    const LeastSquaresSums& moments() const { return moments_; }
    
    const std::vector<Pt>& hull_lo() const { return hull_lo_; }
    
    const std::vector<Pt>& hull_hi() const { return hull_hi_; }

private:
    static double cross(const Pt& o, const Pt& a, const Pt& b) {
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    }

    void push_hull_lo(Pt p) {
        while (hull_lo_.size() >= 2 &&
               cross(hull_lo_[hull_lo_.size() - 2], hull_lo_[hull_lo_.size() - 1], p) >= 0.0) {
            hull_lo_.pop_back();
        }
        hull_lo_.push_back(p);
    }

    void push_hull_hi(Pt p) {
        while (hull_hi_.size() >= 2 &&
               cross(hull_hi_[hull_hi_.size() - 2], hull_hi_[hull_hi_.size() - 1], p) <= 0.0) {
            hull_hi_.pop_back();
        }

        hull_hi_.push_back(p);
    }

    std::size_t count_ = 0;
    Key key_low_ = 0;
    double eps_ = 0.0;
    
    bool coverable_ = false;
    std::size_t forced_cut_ = 0;
    LinearModel model_{ 0.0, 0.0 };
    LeastSquaresSums moments_{};
    
    std::vector<Pt> hull_lo_;
    std::vector<Pt> hull_hi_;
};

}