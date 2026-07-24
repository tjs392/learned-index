/*
* drift_guard.hpp - O(1) drift check, 16 bytes per segment
*
* A segment must know on every mutation whether its model still eps covers
* every key it holds. deviation_i = predict(key_i) - rank_i, keys never move,
* so only rank shifts change deviations, and each op shifts one direction:
*
* insert: ranks at >= r go up so existing deviations fall. The floor decays by
* one, the ceiling can only be raised by the arriving key the caller holds
*
* erase: deviations after r rise. the ceiling decays by one, floor untouched
* decay = 0 when nothing shifted , so sorted stays exact
*
*
*/

#pragma once

#include "status.hpp"

#include <algorithm>

namespace li::detail {

class DriftGuard {
public:

    // tobe call after any refit, rebuild, or bulk load with the exact extreme  deviations
    void reset(double exact_deviation_min, double exact_deviation_max) {
        deviation_floor_ = exact_deviation_min;
        deviation_ceiling_ = exact_deviation_max;
    }

    // arriving deviation is measured against the CURRENT model at the keys new rank
    bool try_absorb_insert(double arriving_deviation, double epsilon, bool later_keys_shifted) {
        const double decay = later_keys_shifted ? 1.0 : 0.0;

        deviation_floor_ = std::min(deviation_floor_ - decay, arriving_deviation);
        deviation_ceiling_ = std::max(deviation_ceiling_, arriving_deviation);

        return holds(epsilon);
    }

    bool try_absorb_erase(double epsilon, bool later_keys_shifted) {
        deviation_ceiling_ += later_keys_shifted ? 1.0 : 0.0;

        return holds(epsilon);
    }

    bool holds(double epsilon) const {
        return deviation_floor_ >= -epsilon && deviation_ceiling_ <= epsilon;
    }

    double deviation_floor() const { return deviation_floor_; }
    double deviation_ceiling() const { return deviation_ceiling_; }


private:

    double deviation_floor_{0.0};
    double deviation_ceiling_{0.0};
};

}