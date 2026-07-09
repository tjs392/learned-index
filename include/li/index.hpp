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
#include <tuple>


namespace li {


struct SegmentDescriptor {
    Key key_low;
    LinearModel model;
    size_t count;

    // TODO: 
    // this is a multithreading artifact - not using in single threaded applications
    // just a reminder to come back and handle this
    uint64_t version = 0;
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

        iterator(const std::vector<detail::PmaBlock>* blocks,
                 std::size_t block, std::size_t slot, Key high)
            : blocks_(blocks), block_(block), slot_(slot), high_(high) {
            settle();
        }

        static iterator make_end(const std::vector<detail::PmaBlock>* blocks) {
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
                const detail::PmaBlock& pma = (*blocks_)[block_];
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

        const std::vector<detail::PmaBlock>* blocks_ = nullptr;
        std::size_t block_ = 0;
        std::size_t slot_  = 0;
        Key high_ = 0;
    };

    RangeView() = default;
    RangeView(const std::vector<detail::PmaBlock>* blocks,
              std::size_t start_block, std::size_t start_slot, Key high)
        : blocks_(blocks), start_block_(start_block),
          start_slot_(start_slot), high_(high) {}

    iterator begin() const { return iterator(blocks_, start_block_, start_slot_, high_); }
    iterator end() const { return iterator::make_end(blocks_); }
    bool empty() const { return begin() == end(); }

private:
    const std::vector<detail::PmaBlock>* blocks_ = nullptr;
    std::size_t start_block_ = 0;
    std::size_t start_slot_  = 0;
    Key high_ = 0;
};

// TODO:
// Don't forget about proactive clean/splitting and optional rebuilding to optimal m
class LearnedIndex {
public:

    LearnedIndex(double epsilon) : epsilon_(epsilon) {}

    void build(std::vector<Key> keys) {
        std::vector<detail::FittedSegment> fitted = detail::segment_stream(keys, epsilon_);

        mapping_table_.reserve(fitted.size());
        blocks_.reserve(fitted.size());

        for (const auto& fit : fitted) {
            mapping_table_.push_back(SegmentDescriptor{
                fit.key_low,
                fit.model,
                fit.count,
            });

            std::vector<Key> slice(
                keys.begin() + static_cast<std::ptrdiff_t>(fit.base_rank),
                keys.begin() + static_cast<std::ptrdiff_t>(fit.base_rank + fit.count)
            );

            std::vector<Payload> payloads(fit.count);
            for (size_t i = 0; i < fit.count; ++i) {
                payloads[i] = static_cast<Payload>(fit.base_rank + i);
            }

            blocks_.push_back(detail::PmaBlock::bulk_load(slice, payloads));
        }

        Rank total_keys = 0;
        for (const auto& d : mapping_table_) {
            LI_ASSERT(d.count > 0);
            total_keys += d.count;
        }
        LI_ASSERT(total_keys == keys.size());
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
    Result<Payload> point_lookup(Key key) const {
        if (mapping_table_.empty()) return Status::not_found;

        const size_t i = find_descriptor(key);
        const SegmentDescriptor& d = mapping_table_[i];

        if (key < d.key_low) return Status::not_found;

        const detail::PmaBlock& pma = blocks_[i];
        if (pma.empty()) return Status::not_found;

        const auto [lo_rank, hi_rank] = local_window(d, pma.size(), key);

        const size_t lo_slot = pma.slot_of_rank(lo_rank);
        const size_t hi_slot = pma.slot_of_rank(hi_rank);

        const std::optional<size_t> hit = pma.find_in(key, lo_slot, hi_slot);
        if (!hit) return Status::not_found;

        return pma.payload_at(*hit);
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

    Status insert(Key key, Payload payload) {
        if (mapping_table_.empty()) {
            const std::vector<Key> seed_key{key};
            const std::vector<Payload> seed_payload{payload};

            blocks_.push_back(detail::PmaBlock::bulk_load(seed_key, seed_payload));
            mapping_table_.push_back(SegmentDescriptor{key, LinearModel{0.0, 0.0}, 1});
            
            return Status::ok;
        }

        const size_t i = find_descriptor(key);

        // TODO: currently i assert non dupes
        // come back this when i implement the uniquifier
        blocks_[i].insert(key, payload);

        auto [dense_keys, dense_payloads, fitted] = refit_segment(i);
        apply_refit(i, dense_keys, dense_payloads, fitted);

        return Status::ok;
    }


    Status erase(Key key) {
        if (mapping_table_.empty()) return Status::not_found;

        const size_t i = find_descriptor(key);
        if (key < mapping_table_[i].key_low) return Status::not_found;

        if (!blocks_[i].erase(key)) return Status::not_found;

        if (blocks_[i].empty()) {
            std::vector<SegmentDescriptor> none;
            std::vector<detail::PmaBlock> none_blocks;

            replace_segments(i, 1, none, none_blocks);
            
            if (i > 0) restore_irreducibility(i - 1, i - 1);
            
            return Status::ok;
        }

        auto [dense_keys, dense_payloads, fitted] = refit_segment(i);
        apply_refit(i, dense_keys, dense_payloads, fitted);
        
        return Status::ok;
    }



    Result<Rank> global_rank_of(Key key) const {
        if (mapping_table_.empty()) return Status::not_found;
        
        const size_t i = find_descriptor(key);
        if (key < mapping_table_[i].key_low) return Status::not_found;
        
        const std::optional<size_t> slot = blocks_[i].find(key);
        if (!slot) return Status::not_found;
        
        return base_rank_of(i) + blocks_[i].local_rank(*slot);
    }

    const std::vector<detail::PmaBlock>& blocks_for_test() const { return blocks_; }
    

private:

    std::vector<SegmentDescriptor> mapping_table_;
    std::vector<detail::PmaBlock> blocks_;

    double epsilon_;

    //TODO:
    // defaulting this to 0.0 to hold models to perfect epsilon
    // play around with this, maybe grid search the optimal merge slack param or sumthn
    // in da future doe
    double merge_slack_ = 0.0;

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


    // TODO:
    // this is O(n) prefex scan, but it's off the hot path, so only fires on rank query
    // we can do a fenwick maybe, but just logging todo here fo rthe future. probably won't matter
    Rank base_rank_of(size_t segment_index) const {
        Rank preceding_key_count = 0;
        for (size_t earlier = 0; earlier < segment_index; ++earlier) {
            preceding_key_count += mapping_table_[earlier].count;
        }
        return preceding_key_count;
    }

    // TODO:
    // Remove when hull tree is implemented
    // right now this is O(n) it does the dumping of gapped -> dense, and 
    // then runs the frozen cone from scratch
    // after hull tree, i can query the maintained hull
    // thisll be O(log w) feasibility check and the O(log^2w) split
    // for now just working insert with segment refitting to keep optimal stuff
    std::tuple<std::vector<Key>, std::vector<Payload>, std::vector<detail::FittedSegment>> refit_segment(size_t segment_index) {
        auto [dense_keys, dense_payloads] = blocks_[segment_index].dump_sorted();
        std::vector<detail::FittedSegment> fitted = detail::segment_stream(dense_keys, epsilon_);
        
        return std::make_tuple(std::move(dense_keys), std::move(dense_payloads), std::move(fitted));
    }

    // TODO:
    // this is O(n). which is fine for now since m << N and cache resident.
    // we can move this to O(logn) by using a tree-d mapping table
    // but we can measure the performance of this... might not be needed cause
    // this table is pretty cache freidnyl
    void replace_segments(size_t first, size_t remove_count,
                          std::vector<SegmentDescriptor>& new_descriptors,
                          std::vector<detail::PmaBlock>& new_blocks) {
        const auto at = static_cast<std::ptrdiff_t>(first);
        const auto rm = static_cast<std::ptrdiff_t>(remove_count);

        mapping_table_.erase(mapping_table_.begin() + at, mapping_table_.begin() + at + rm);
        mapping_table_.insert(mapping_table_.begin() + at,
                              new_descriptors.begin(), new_descriptors.end());

        blocks_.erase(blocks_.begin() + at, blocks_.begin() + at + rm);
        blocks_.insert(blocks_.begin() + at,
                       std::make_move_iterator(new_blocks.begin()),
                       std::make_move_iterator(new_blocks.end()));
    }

    // TODO: 
    // Currently I am deferring the "which "
    void apply_refit(size_t i,
                     std::vector<Key>& dense_keys,
                     std::vector<Payload>& dense_payloads,
                     std::vector<detail::FittedSegment>& fitted) {
        if (fitted.size() == 1) {
            SegmentDescriptor& d = mapping_table_[i];

            d.key_low = fitted[0].key_low;
            d.model = fitted[0].model;
            d.count = fitted[0].count;

            restore_irreducibility(i, i);
            return;
        }

        std::vector<SegmentDescriptor> new_descriptors;
        std::vector<detail::PmaBlock> new_blocks;

        new_descriptors.reserve(fitted.size());
        new_blocks.reserve(fitted.size());

        size_t slice_start = 0;
        for (const auto& fit : fitted) {
            new_descriptors.push_back(SegmentDescriptor{fit.key_low, fit.model, fit.count});
            new_blocks.push_back(detail::PmaBlock::bulk_load(
                std::span<const Key>(dense_keys.data() + slice_start, fit.count),
                std::span<const Payload>(dense_payloads.data() + slice_start, fit.count)
            ));
            slice_start += fit.count;
        }

        const size_t k = fitted.size();

        replace_segments(i, 1, new_descriptors, new_blocks);
        restore_irreducibility(i, i + k - 1);
    }

    // TODO: hull tree 
    // right now this just dumps both blocks dense + reruns the cone
    // O(w) this is just correct. once hull tree hits, we can do
    // the bridge merge whichll be O(logw). no dumping or rescan, etc.
    // just swap to seg stream line EASY :D
    bool try_merge(size_t a) {
        auto [key_a, payload_a] = blocks_[a].dump_sorted();
        auto [key_b, payload_b] = blocks_[a + 1].dump_sorted();

        std::vector<Key> union_keys;
        union_keys.reserve(key_a.size() + key_b.size());
        union_keys.insert(union_keys.end(), key_a.begin(), key_a.end());
        union_keys.insert(union_keys.end(), key_b.begin(), key_b.end());

        auto fitted = detail::segment_stream(union_keys, epsilon_ * (1.0 - merge_slack_));
        if (fitted.size() != 1) return false;

        std::vector<Payload> union_payloads;
        union_payloads.reserve(payload_a.size() + payload_b.size());
        union_payloads.insert(union_payloads.end(), payload_a.begin(), payload_a.end());
        union_payloads.insert(union_payloads.end(), payload_b.begin(), payload_b.end());

        std::vector<SegmentDescriptor> merged{
            SegmentDescriptor{key_a.front(), fitted[0].model, key_a.size() + key_b.size()}
        };
        
        std::vector<detail::PmaBlock> merged_block;
        merged_block.push_back(detail::PmaBlock::bulk_load(union_keys, union_payloads));

        replace_segments(a, 2, merged, merged_block);
        return true;
    }

    // TODO:
    // right now i am just saying, greedy merge the right segment.
    // but what if merging with the left segment or something in between is better?
    // pick some heuristic/cost model to determine where to merge MAYBE
    // but this might cause overhead, so just try this out when optimizing
    void restore_irreducibility(size_t lo, size_t hi) {
        size_t merges = 0;
        
        // right boundary absorbs the run's right neighbor, 
        // so the merged segment stays at the index hi. so recheck it against
        // its new neighbor after it was merge
        while (hi + 1 < mapping_table_.size() && try_merge(hi)) ++merges;

        // similar logic here, left absorbs its left neighbor, so need to descrement
        while (lo > 0 && try_merge(lo - 1)) { ++merges; --lo; }
        
        LI_ASSERT(merges <= 2);
    }

};



}