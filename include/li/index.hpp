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

class RangeView {
public:
    // Custom iterate for PMA gapped arrays (currently just works like a normal iterator)
    class iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type        = Key;
        using difference_type    = std::ptrdiff_t;
        using pointer           = const Key*;
        using reference         = const Key&;

        explicit iterator(const Key* p) : p_(p) {}

        reference operator*() const { return *p_; } 
        iterator& operator++() { ++p_; return *this; }

        bool operator!=(const iterator& o) const { return p_ != o.p_; }
        bool operator==(const iterator& o) const { return p_ == o.p_; }
    
    private:
        const Key* p_;
    };

    RangeView(const Key* b, const Key* e) : begin_(b), end_(e) {}

    iterator begin() const { return iterator(begin_); }
    iterator end() const { return iterator(end_); }
    bool empty() const { return begin_ == end_; }
    size_t size() const { return static_cast<size_t>(end_ - begin_); }

private:
    const Key* begin_;
    const Key* end_;
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

        auto [lo, hi] = search_window(key);
        const auto found = std::lower_bound(keys_.begin() + lo, keys_.begin() + hi, key);

        if (found != keys_.begin() + hi && *found == key) {
            return static_cast<uint64_t>(found - keys_.begin());
        }

        return Status::not_found;
    }

    // TODO: no model narrowing, just plain binary search
    // this works for now  but in the future need to implement model narrowing steps
    // O(logn) -> O(logepsilon)
    RangeView range_lookup(Key low, Key high) const {
        if (keys_.empty() || low > high) {
            return RangeView(keys_.data(), keys_.data());
        }

        auto start = std::lower_bound(keys_.begin(), keys_.end(), low);
        auto end = std::upper_bound(keys_.begin(), keys_.end(), high);

        const uint64_t si = static_cast<uint64_t>(start - keys_.begin());
        const uint64_t ei = static_cast<uint64_t>(end - keys_.begin());

        if (si >= ei) {
            return RangeView(keys_.data(), keys_.data());
        }

        return RangeView(keys_.data() + si, keys_.data() + ei);
    }

    std::pair<uint64_t, uint64_t> search_window(Key key) const {
        const SegmentDescriptor& d = mapping_table_[find_descriptor(key)];

        const Rank local_pred = predict(d.model, key, d.key_low);

        const uint64_t eps_ceil = static_cast<uint64_t>(std::ceil(epsilon_));
        const uint64_t low  = (local_pred > eps_ceil) ? (local_pred - eps_ceil) : 0;

        uint64_t high = local_pred + eps_ceil + 1;

        if (high > d.count) high = d.count;
        
        return { d.base_rank + low, d.base_rank + high };
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