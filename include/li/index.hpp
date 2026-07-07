/*
* index.hpp - The index
*/

#pragma once

#include "types.hpp"
#include "model.hpp"
#include "status.hpp"
#include "segmentation.hpp"
#include "pma.hpp"

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

        mapping_table_.reserve(specs.size());
        blocks_.reserve(specs.size());

        for (const auto& spec : specs) {
            mapping_table_.push_back(SegmentDescriptor {
                spec.key_low,
                spec.model,
                spec.base_rank,
                spec.count,
            });

            std::vector<Key> slice(
                keys.begin() + static_cast<std::ptrdiff_t>(spec.base_rank),
                keys.begin() + static_cast<std::ptrdiff_t>(spec.base_rank + spec.count)
            );

            std::vector<Rank> payloads(spec.count);

            for (size_t i = 0; i < spec.count; ++i) {
                payloads[i] = static_cast<Rank>(spec.base_rank + i);
            }

            blocks_.push_back(detail::PmaBlock<Rank>::bulk_load(slice, payloads));
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
        if (mapping_table_.empty()) return Status::not_found;

        const size_t i = find_descriptor(key);
        const SegmentDescriptor& d = mapping_table_[i];

        if (key < d.key_low) return Status::not_found;

        const detail::PmaBlock<Rank>& pma = blocks_[i];
        if (pma.empty()) return Status::not_found;

        const auto [lo_rank, hi_rank] = local_window(d, pma.size(), key);

        const size_t lo_slot = pma.slot_of_rank(lo_rank);
        const size_t hi_slot = pma.slot_of_rank(hi_rank);

        const std::optional<size_t> hit = pma.find_in(key, lo_slot, hi_slot);
        if (!hit) return Status::not_found;

        return static_cast<uint64_t>(pma.payload_at(*hit));
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
    std::vector<detail::PmaBlock<Rank>> blocks_;
    
    // TODO: this assumes keys are stricly increasing
    // Duplicate user keys are assumed to be disambiguated upstream by the uniquifier
    // Until that's implemented, no duplicates is a preocndition
    std::vector<Key> keys_;

    double epsilon_;

    std::pair<Rank, Rank> local_window(const SegmentDescriptor& d, size_t count, Key key) const {
        Rank pred = predict(d.model, key, d.key_low);

        const Rank max_rank = static_cast<Rank>(count - 1);
        if (pred > max_rank) pred = max_rank;

        const Rank eps_ceil = static_cast<Rank>(std::ceil(epsilon_));

        const Rank lo_rank = (pred > eps_ceil) ? (pred - eps_ceil) : Rank{0};
        Rank hi_rank = pred + eps_ceil + 1;
        if (hi_rank > max_rank) hi_rank = max_rank;

        return { lo_rank, hi_rank };
    }
};


}