#pragma once

#include "types.hpp"
#include "model.hpp"
#include "status.hpp"
#include "segmentation.hpp"

#include <vector>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <limits>


namespace li {


struct SegmentDescriptor {
    Key key_low;
    Model model;
    uint64_t base_rank;
    size_t count;
};


class Index {

public:

    Index(double epsilon) : epsilon_(epsilon) {}

    // TODO: base_rank is currently the running offset into the flat sorted keys_ 
    // When i implement PMA, a segment's pos is derived from its physical block location

    void build(std::vector<Key> keys) {
        std::vector<detail::SegmentSpec> specs = detail::segment_stream(keys, epsilon_);
        for (const auto& spec : specs) {
            mapping_table_.push_back(SegmentDescriptor{
                spec.key_low,
                spec.model,
                spec.base_rank,
                spec.count,
            });
        }

        uint64_t expected_base = 0;
        for (const auto& d : mapping_table_) {
            LI_ASSERT(d.base_rank == expected_base);
            LI_ASSERT(d.count > 0);
            expected_base += d.count;
        }
        LI_ASSERT(expected_base == keys.size());

        keys_ = std::move(keys);
    }

    // TODO: this search over the mapping table is correct
    // routing is a measured speedup layer on top later
    // benchmark l8r

    // Returns index into mapping_table_
    size_t find_descriptor(Key key) const {
        LI_ASSERT(!mapping_table_.empty());
        auto it = std::upper_bound(mapping_table_.begin(), mapping_table_.end(), key, 
            [](Key k, const SegmentDescriptor& d) {
                return k < d.key_low;
            }
        );

        if (it == mapping_table_.begin()) {
            return 0;
        }

        return static_cast<size_t>((it - mapping_table_.begin()) - 1);
        
    }

    // TODO:
    // clamp the search window here at lookup time
    // TODO: last mile search is plain binary search over the winows. this is
    // correct, but for future optimizations we'll want branch free/simd/prefetch stuff
    // TODO: Handle duplicates down the line

    // Returns the first match it finds
    Result<uint64_t> point_lookup(Key key) const {
        if (keys_.empty()) return Status::not_found;

        const SegmentDescriptor& d = mapping_table_[find_descriptor(key)];
        
        if (key < d.key_low) return Status::not_found;
        
        const Pos local_prediction = predict(d.model, key, d.key_low);

        const uint64_t eps_ceil = static_cast<uint64_t>(std::ceil(epsilon_));
        const uint64_t low = (local_prediction > eps_ceil) ? (local_prediction - eps_ceil) : 0;

        uint64_t high = local_prediction + eps_ceil + 1;
        if (high > d.count) high = d.count;

        const auto slice_begin = keys_.begin() + d.base_rank + low;
        const auto slice_end = keys_.begin() + d.base_rank + high;
        const auto found = std::lower_bound(slice_begin, slice_end, key);
        if (found != slice_end && *found == key) {
            return static_cast<uint64_t>(found - keys_.begin());
        }
        
        return Status::not_found;
    }

    const std::vector<SegmentDescriptor>& mapping_table_for_test() const {
        return mapping_table_;
    }


private:

    std::vector<SegmentDescriptor> mapping_table_;
    
    // TODO: this assumes keys are stricly increasing
    // Duplicate user keys are assumed to be disambiguated upstream by the uniquifier
    // Until that's implemented, no duplicates is a preocndition
    std::vector<Key> keys_;

    double epsilon_;

};


}