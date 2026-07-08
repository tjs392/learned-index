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
#include <iterator>


namespace li {


struct SegmentDescriptor {
    Key key_low;
    LinearModel model;
    uint64_t base_rank;
    size_t count;
};

class RangeView {
public:
    // Custom iterate for PMA gapped arrays (currently just works like a normal iterator)
    class iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type        = Key;
        using difference_type   = std::ptrdiff_t;
        using pointer           = const Key*;
        using reference         = Key;

        iterator(const std::vector<detail::PmaBlock<Rank>>* blocks,
                 std::size_t block, std::size_t slot, Key high)
            : blocks_(blocks), block_(block), slot_(slot), high_(high) {
            settle();
        }

        static iterator make_end(const std::vector<detail::PmaBlock<Rank>>* blocks) {
            iterator it;
            it.blocks_ = blocks;
            it.block_  = blocks ? blocks->size() : 0;
            return it;
        }

        Key operator*() const { return (*blocks_)[block_].key_at(slot_); }

        iterator& operator++() {
            ++slot_;
            settle();
            return *this;
        }

        bool operator==(const iterator& o) const {
            const bool ae = at_end(), be = o.at_end();
            if (ae || be) return ae && be;
            return block_ == o.block_ && slot_ == o.slot_;
        }
        bool operator!=(const iterator& o) const { return !(*this == o); }

    private:
        iterator() = default;

        bool at_end() const { return !blocks_ || block_ >= blocks_->size(); }

        void settle() {
            while (blocks_ && block_ < blocks_->size()) {
                const detail::PmaBlock<Rank>& pma = (*blocks_)[block_];
                std::size_t s = pma.next_occupied(slot_);
                if (s < pma.capacity()) {
                    slot_ = s;
                    if (pma.key_at(slot_) > high_) block_ = blocks_->size();
                    return;
                }
                ++block_;
                slot_ = 0;
            }
        }

        const std::vector<detail::PmaBlock<Rank>>* blocks_ = nullptr;
        std::size_t block_ = 0;
        std::size_t slot_  = 0;
        Key high_ = 0;
    };

    RangeView() = default;
    RangeView(const std::vector<detail::PmaBlock<Rank>>* blocks,
              std::size_t start_block, std::size_t start_slot, Key high)
        : blocks_(blocks), start_block_(start_block),
          start_slot_(start_slot), high_(high) {}

    iterator begin() const { return iterator(blocks_, start_block_, start_slot_, high_); }
    iterator end() const { return iterator::make_end(blocks_); }
    bool empty() const { return begin() == end(); }

private:
    const std::vector<detail::PmaBlock<Rank>>* blocks_ = nullptr;
    std::size_t start_block_ = 0;
    std::size_t start_slot_  = 0;
    Key high_ = 0;
};

class LearnedIndex {
public:

    LearnedIndex(double epsilon) : epsilon_(epsilon) {}

    void build(std::vector<Key> keys) {
        std::vector<detail::FittedSegment> specs = detail::segment_stream(keys, epsilon_);

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
        if (mapping_table_.empty() || low > high) {
            return RangeView(&blocks_, blocks_.size(), 0, high);
        }
        const size_t start_bi = find_descriptor(low);
        const std::size_t start_slot = blocks_[start_bi].lower_bound(low);
        return RangeView(&blocks_, start_bi, start_slot, high);
    }

    const std::vector<SegmentDescriptor>& mapping_table_for_test() const {
        return mapping_table_;
    }


private:

    std::vector<SegmentDescriptor> mapping_table_;
    std::vector<detail::PmaBlock<Rank>> blocks_;

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