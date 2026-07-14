/*
* cedar_index.hpp - The learned index
*
* One mapping table of Segments, each a model plus an owned PmaBlock and EpsilonToleranceBand.
* every insert and delete runs the cheap band feasibility check first and only falls back to the exact
* O(W_s) cone (minimal_line_cover) on a break
* Segments are capped at W_s keys and we do cap split before overflow
* merged when a neighbor pair fits under W_m, and each carries a refit budget B that
* force splits instead of thrashing
* Important: no cascade since theres a bounded number of structural ops per operation.
*/

#pragma once

#include "types.hpp"
#include "model.hpp"
#include "status.hpp"
#include "pma.hpp"
#include "segmentation.hpp"
#include "epsilon_tolerance_band.hpp"
#include "minimal_line_cover.hpp"

#include <vector>
#include <span>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace li {

struct Segment {
    Key key_low;
    std::size_t count;
    LinearModel model;
    detail::PmaBlock pma;
    detail::EpsilonToleranceBand tolerance_band;
    int refit_budget;

    // TODO: concurrency primitive
    std::uint64_t version;

    std::size_t size() const { return count; }
};

class CedarIndex {
public:
    // w_s is the split cap: a segment is never allowed to have > w_s keys
    // w_m is the merge threshold: two adjacent segments only combine when their combined count is <= w_m
    // TODO: right now 2048 is just cache-conscious; later we can grid search these for optimal parameters (keep in mind this is cpu dependent)
    // b is the refit budget: this forces a split when the segment is refitting too much
    CedarIndex(double epsilon, std::size_t w_s = 2048, std::size_t w_m = 1024, int b = 11)
        : epsilon_(epsilon), w_s_(w_s), w_m_(w_m), b_(b),
          max_capacity_slots_(detail::PmaBlock::capacity_for_key_cap(w_s)) {}

    void build(const std::vector<Key>& keys) {
        mapping_table_.clear();
        if (keys.empty()) return;

        std::vector<detail::FittedSegment> specs = detail::segment_stream(keys, epsilon_);

        for (const auto& spec : specs) {
            std::size_t offset = 0;
            while (offset < spec.count) {
                std::size_t take = std::min(spec.count - offset, w_s_);

                std::span<const Key> chunk_keys(keys.data() + spec.base_rank + offset, take);
                std::vector<Payload> payloads(take);
                for (std::size_t j = 0; j < take; ++j)
                    payloads[j] = Payload(spec.base_rank + offset + j);

                if (take == spec.count) {
                    mapping_table_.push_back(
                        make_segment_with_model(chunk_keys, payloads, spec.model));
                } else {
                    mapping_table_.push_back(make_segment(chunk_keys, payloads));
                }

                offset += take;
            }
        }
    }


    Status insert(Key key, Payload payload) {
        last_structural_ops_ = 0;

        if (mapping_table_.empty()) {
            std::vector<Key> one_key{ key };
            std::vector<Payload> one_payload{ payload };
            mapping_table_.push_back(make_segment(one_key, one_payload));
            return Status::ok;
        }

        std::size_t index = route(key);
        Segment& segment = mapping_table_[index];

        LI_ASSERT(segment.pma.find(key) == std::nullopt);

        bool front_insert = key < segment.key_low;

        detail::PmaBlock::EditResult edit =
            (segment.pma.empty() || key > segment.pma.max_key())
                ? segment.pma.append(key, payload)
                : segment.pma.insert(key, payload);
        segment.count += 1;

        if (front_insert) {
            Key old_low = segment.key_low;
            segment.model.beta += 1.0 - segment.model.alpha * static_cast<double>(old_low - key);
            segment.key_low = key;

            if (!edit.moved_empty()) {
                segment.tolerance_band.rebuild_range(edit.moved_lo, edit.moved_hi,
                                                     segment.pma, segment.model, segment.key_low);
            } else {
                double dev = line_at(segment.model, key, segment.key_low)
                             - double(segment.pma.local_rank(edit.slot));
                segment.tolerance_band.place(edit.slot, dev);
            }

        } else {
            sync_band_after_insert(index, key, edit);
        }

        double deviation_low = segment.tolerance_band.min_over_occupied();
        double deviation_high = segment.tolerance_band.max_over_occupied();
        bool in_band = deviation_low >= -epsilon_ && deviation_high <= epsilon_;

        if (in_band) {
            if (segment.count <= w_s_) {
                segment.version += 1;
                // even in band inserts cause rank shifts and can change merging feasibility
                last_structural_ops_ = try_merge_flanks(index, index + 1);
                return Status::ok;
            }
            int ops = apply_middle_split(index);
            ops += try_merge_flanks(index, index + 2);
            last_structural_ops_ = ops;
            return Status::ok;
        }

        last_structural_ops_ = rebuild_broken_segment(index);
        return Status::ok;
    }

    Status erase(Key key) {
        last_structural_ops_ = 0;

        if (mapping_table_.empty()) return Status::not_found;

        std::size_t index = route(key);
        Segment& segment = mapping_table_[index];
        if (key < segment.key_low) return Status::not_found;

        detail::PmaBlock::EditResult edit = segment.pma.erase(key);
        if (!edit.found) return Status::not_found;
        segment.count -= 1;

        if (segment.count == 0) {
            remove_segment(index);
            int ops = 1;
            if (index > 0 && index < mapping_table_.size())
                if (try_merge_pair(index - 1)) ops += 1;
            last_structural_ops_ = ops;
            return Status::ok;
        }

        sync_band_after_erase(index, edit);

        double deviation_low = segment.tolerance_band.min_over_occupied();
        double deviation_high = segment.tolerance_band.max_over_occupied();
        bool in_band = deviation_low >= -epsilon_ && deviation_high <= epsilon_;

        int ops = 0;
        if (!in_band) {
            ops += rebuild_broken_segment(index);
        } else {
            ops += try_merge_flanks(index, index + 1);
            segment_at(index).version += 1;
        }

        last_structural_ops_ = ops;
        return Status::ok;
    }

    Result<Payload> point_lookup(Key key) const {
        if (mapping_table_.empty()) return Status::not_found;

        std::size_t index = route(key);
        const Segment& segment = mapping_table_[index];
        if (key < segment.key_low) return Status::not_found;
        if (segment.pma.empty()) return Status::not_found;

        const auto [lo_rank, hi_rank] = local_window(segment, key);
        std::size_t lo_slot = segment.pma.slot_of_rank(lo_rank);
        std::size_t hi_slot = segment.pma.slot_of_rank(hi_rank);

        std::optional<std::size_t> hit = segment.pma.find_in(key, lo_slot, hi_slot);
        if (!hit) return Status::not_found;
        return segment.pma.payload_at(*hit);
    }


    // TODO(perf): materializes into a vector; a streaming callback / lazy view would avoid the
    // allocation for scan heavy workloads and would be cache friendly
    // adding payloads to the output is a single line extension when a benchmark needs value scans.
    std::vector<Key> range_lookup(Key lo, Key hi) const {
        std::vector<Key> out;
        if (hi < lo || mapping_table_.empty()) return out;

        for (std::size_t index = route(lo); index < mapping_table_.size(); ++index) {
            const Segment& segment = mapping_table_[index];
            if (segment.key_low > hi) break;

            for (std::size_t slot = segment.pma.next_occupied(0); slot < segment.pma.capacity();
                 slot = segment.pma.next_occupied(slot + 1)) {
                Key k = segment.pma.key_at(slot);
                if (k < lo) continue;
                if (k > hi) return out;
                out.push_back(k);
            }
        }

        return out;
    }

    Result<Rank> global_rank_of(Key key) const {
        if (mapping_table_.empty()) return Status::not_found;

        std::size_t index = route(key);
        const Segment& segment = mapping_table_[index];
        if (key < segment.key_low) return Status::not_found;

        std::optional<std::size_t> slot = segment.pma.find(key);
        if (!slot) return Status::not_found;
        return base_rank(index) + segment.pma.local_rank(*slot);
    }

    const std::vector<Segment>& segments_for_test() const { return mapping_table_; }
    int last_structural_ops() const { return last_structural_ops_; }
    std::size_t cover_recomputes() const { return cover_recomputes_; }
    std::size_t num_segments() const { return mapping_table_.size(); }
    double epsilon() const { return epsilon_; }
    std::size_t w_s() const { return w_s_; }
    std::size_t w_m() const { return w_m_; }

private:
    std::vector<Segment> mapping_table_;
    double epsilon_;
    std::size_t w_s_;
    std::size_t w_m_;
    int b_;
    std::size_t max_capacity_slots_;
    int last_structural_ops_ = 0;
    std::size_t cover_recomputes_ = 0;

    Segment& segment_at(std::size_t index) { return mapping_table_[index]; }

    // TODO:
    // Riyte us stukk just a binary search over mapping table
    std::size_t route(Key key) const {
        auto it = std::upper_bound(mapping_table_.begin(), mapping_table_.end(), key,
            [](Key k, const Segment& s) { return k < s.key_low; });
        if (it == mapping_table_.begin()) return 0;

        return std::size_t((it - mapping_table_.begin()) - 1);
    }

    // TODO(perf):
    // probably can bitscan this for preceding?
    // idk maybe need bitmap. but can test
    Rank base_rank(std::size_t index) const {
        Rank preceding = 0;
        for (std::size_t j = 0; j < index; ++j) preceding += mapping_table_[j].count;
        return preceding;
    }

    std::pair<Rank, Rank> local_window(const Segment& segment, Key key) const {
        Rank pred = predict(segment.model, key, segment.key_low);
        Rank max_rank = Rank(segment.count - 1);
        if (pred > max_rank) pred = max_rank;

        Rank eps_ceil = Rank(std::ceil(epsilon_));
        Rank lo_rank = pred > eps_ceil ? pred - eps_ceil : Rank{ 0 };
        Rank hi_rank = pred + eps_ceil + 1;
        if (hi_rank > max_rank) hi_rank = max_rank;

        return { lo_rank, hi_rank };
    }

    Segment make_segment_with_model(std::span<const Key> keys,
                                    std::span<const Payload> payloads, 
                                    LinearModel model) {
        Segment segment;
        segment.key_low = keys[0];
        segment.count = keys.size();
        segment.model = model;
        segment.pma = detail::PmaBlock::bulk_load(keys, payloads, max_capacity_slots_);
        segment.tolerance_band =
            detail::EpsilonToleranceBand::build(segment.pma, model, segment.key_low);
        segment.refit_budget = b_;
        segment.version = 0;

        return segment;
    }

    Segment make_segment(std::span<const Key> keys, std::span<const Payload> payloads) {
        detail::LineCoverResult verdict = detail::minimal_line_cover(keys, epsilon_);
        LI_ASSERT(verdict.status == detail::LineCoverStatus::COVERABLE);

        return make_segment_with_model(keys, payloads, verdict.model);
    }

    void sync_band_after_insert(std::size_t index, 
                                Key key,
                                const detail::PmaBlock::EditResult& edit) {
        Segment& segment = mapping_table_[index];

        if (!edit.moved_empty()) {
            segment.tolerance_band.rebuild_range(edit.moved_lo, edit.moved_hi,
                                                 segment.pma, segment.model, segment.key_low);
            segment.tolerance_band.add_to_range(edit.moved_hi, segment.pma.capacity(), -1.0);
        } else {
            segment.tolerance_band.add_to_range(edit.slot + 1, segment.pma.capacity(), -1.0);
            double deviation = line_at(segment.model, key, segment.key_low)
                               - double(segment.pma.local_rank(edit.slot));
            segment.tolerance_band.place(edit.slot, deviation);
        }
    }

    void sync_band_after_erase(std::size_t index, const detail::PmaBlock::EditResult& edit) {
        Segment& segment = mapping_table_[index];

        if (!edit.moved_empty()) {
            segment.tolerance_band.rebuild_range(edit.moved_lo, edit.moved_hi,
                                                 segment.pma, segment.model, segment.key_low);
        } else {
            segment.tolerance_band.clear(edit.slot);
            segment.tolerance_band.add_to_range(edit.slot, segment.pma.capacity(), 1.0);
        }
    }

    // IMPORTANT
    // This is the slow break path, activated when the current model no longer covers the segment
    // Need to ask the solver then refit or split and try to merge segs
    int rebuild_broken_segment(std::size_t index) {
        auto [keys, payloads] = mapping_table_[index].pma.dump_sorted();
        detail::LineCoverResult verdict = detail::minimal_line_cover(keys, epsilon_);

        int ops = 0;
        ++cover_recomputes_;

        if (verdict.status == detail::LineCoverStatus::COVERABLE) {
            bool force_split = mapping_table_[index].refit_budget <= 0
                               && mapping_table_[index].count > w_m_;

            if (!force_split) {
                apply_refit(index, verdict.model, keys[0]);
                if (mapping_table_[index].count > w_s_) {
                    ops += apply_middle_split(index);
                    ops += try_merge_flanks(index, index + 2);
                    return ops;
                }

                ops += try_merge_flanks(index, index + 1);
                return ops;
            }

            // see logic on segment cap force splits above
            std::size_t k = std::size_t(apply_middle_split(index));
            ops += int(k);
            ops += try_merge_flanks(index, index + k);
            return ops;
        }

        std::size_t k = apply_piece_split(index, verdict.pieces, keys, payloads);
        ops += int(k);
        ops += try_merge_flanks(index, index + k);
        return ops;
    }

    void apply_refit(std::size_t index, LinearModel new_model, Key new_key_low) {
        Segment& segment = mapping_table_[index];
        segment.key_low = new_key_low;
        segment.model = new_model;
        segment.tolerance_band.rebuild_all(segment.pma, new_model, new_key_low);
        segment.refit_budget -= 1;
        segment.version += 1;
    }

    int apply_middle_split(std::size_t index) {
        auto [keys, payloads] = mapping_table_[index].pma.dump_sorted();
        LI_ASSERT(keys.size() >= 2);

        std::size_t mid = keys.size() / 2;

        std::vector<Segment> halves;
        halves.push_back(make_segment(std::span<const Key>(keys.data(), mid),
                                      std::span<const Payload>(payloads.data(), mid)));
        halves.push_back(make_segment(std::span<const Key>(keys.data() + mid, keys.size() - mid),
                                      std::span<const Payload>(payloads.data() + mid, keys.size() - mid)));

        replace_range(index, 1, std::move(halves));
        return 2;
    }

    std::size_t apply_piece_split(std::size_t index,
                                  const std::vector<detail::LineCoverPiece>& pieces,
                                  const std::vector<Key>& keys,
                                  const std::vector<Payload>& payloads) {
        std::vector<Segment> parts;
        parts.reserve(pieces.size());
        for (const auto& piece : pieces) {
            std::size_t len = piece.end - piece.begin;
            parts.push_back(make_segment_with_model(
                std::span<const Key>(keys.data() + piece.begin, len),
                std::span<const Payload>(payloads.data() + piece.begin, len),
                piece.model));
        }

        std::size_t k = parts.size();
        replace_range(index, 1, std::move(parts));
        return k;
    }


    // only left and right are provably mergable after split
    // irreducibility lemma
    int try_merge_flanks(std::size_t lo, std::size_t hi) {
        int merges = 0;
        if (hi < mapping_table_.size())
            if (try_merge_pair(hi - 1)) merges += 1;
        if (lo > 0)
            if (try_merge_pair(lo - 1)) merges += 1;
        return merges;
    }

    bool try_merge_pair(std::size_t left_index) {
        if (left_index + 1 >= mapping_table_.size()) return false;

        Segment& left = mapping_table_[left_index];
        Segment& right = mapping_table_[left_index + 1];
        if (left.count + right.count > w_m_) return false;

        auto [left_keys, left_payloads] = left.pma.dump_sorted();
        auto [right_keys, right_payloads] = right.pma.dump_sorted();

        std::vector<Key> keys;
        keys.reserve(left_keys.size() + right_keys.size());
        keys.insert(keys.end(), left_keys.begin(), left_keys.end());
        keys.insert(keys.end(), right_keys.begin(), right_keys.end());

        detail::LineCoverResult verdict = detail::minimal_line_cover(keys, epsilon_);
        if (verdict.status != detail::LineCoverStatus::COVERABLE) return false;

        std::vector<Payload> payloads;
        payloads.reserve(left_payloads.size() + right_payloads.size());
        payloads.insert(payloads.end(), left_payloads.begin(), left_payloads.end());
        payloads.insert(payloads.end(), right_payloads.begin(), right_payloads.end());

        Segment merged = make_segment_with_model(keys, payloads, verdict.model);
        replace_one(left_index, 2, std::move(merged));
        return true;
    }

    void remove_segment(std::size_t index) {
        mapping_table_.erase(mapping_table_.begin() + std::ptrdiff_t(index));
    }

    void replace_range(std::size_t first, std::size_t remove_count, std::vector<Segment>&& news) {
        auto at = std::ptrdiff_t(first);
        auto rm = std::ptrdiff_t(remove_count);
        mapping_table_.erase(mapping_table_.begin() + at, mapping_table_.begin() + at + rm);
        mapping_table_.insert(mapping_table_.begin() + at,
                              std::make_move_iterator(news.begin()),
                              std::make_move_iterator(news.end()));
    }

    void replace_one(std::size_t first, std::size_t remove_count, Segment&& one) {
        auto at = std::ptrdiff_t(first);
        auto rm = std::ptrdiff_t(remove_count);
        mapping_table_.erase(mapping_table_.begin() + at, mapping_table_.begin() + at + rm);
        mapping_table_.insert(mapping_table_.begin() + at, std::move(one));
    }
};

}