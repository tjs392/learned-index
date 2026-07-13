/*
* epsilon_tolerance_band.hpp - Per segment feasibility maintained under edits

* A flat array embedded lazy segment tree over one PmaBlock's slots. 
* each occupied its signed deviation from the segment's model line in ranks
*
* deviation(slot) = line_at(model, key[slot], key_low) - local_rank(slot)
*
* The segment is within tolerance iff every stored deviation lies in [-epsilon, +epsilon]
*
* This is SLOT INDEXED not RANK INDEXED
*
* O(log W_s) place/clear/add_to_range
* O(q) feasibility read;
* O(window + log W_s) rebuild range after rebalance
* O(W_s) rebuild_all afer refit
* No ponters, bounded depth, etc.
* 
*/

#pragma once

#include "types.hpp"
#include "model.hpp"
#include "pma.hpp"
#include "status.hpp"

#include <vector>
#include <limits>
#include <algorithm>
#include <cstddef>

namespace li::detail {

// The EpsilonToleranceBand answers "is every key in this seg still within epsilon of where the model predicts"
// And answers it fast on the hot path
class EpsilonToleranceBand {
public:
    // TODO(perf): this runs on every make_segment, so every split, merge and refit
    // each call heap allocates + zero fills 2* cap Nodes
    // ~32 bytes each
    // is W_s=2048 at .7 density, that's like ~6000 nodes, ~185KB.
    //  1: reuse a per thread node buffer instead of a fresh alloc per band
    //  2: window leaf variant
    static EpsilonToleranceBand build(const PmaBlock& pma, LinearModel model, Key key_low) {
        EpsilonToleranceBand band;

        band.leaf_count_ = pma.capacity();
        band.nodes_.assign(2 * band.leaf_count_, Node{});
        band.full_rebuild(pma, model, key_low);

        return band;
    }

    void place(std::size_t slot, double deviation) {
        LI_ASSERT(slot < leaf_count_);
        point_update(kRoot, 0, leaf_count_, slot, true, deviation);
    }

    void clear(std::size_t slot) {
        LI_ASSERT(slot < leaf_count_);
        point_update(kRoot, 0, leaf_count_, slot, false, 0.0);
    }

    // Lazily shift every occupied slot in [slot_lo, slot_hi) by delta. The suffix rank shift
    // after an interior insert is add_to_range(.., -1); after a delete, +1.
    void add_to_range(std::size_t slot_lo, std::size_t slot_hi, double delta) {
        if (slot_hi > leaf_count_) slot_hi = leaf_count_;
        if (slot_lo >= slot_hi) return;

        range_add(kRoot, 0, leaf_count_, slot_lo, slot_hi, delta);
    }

    double min_over_occupied() const { return nodes_[kRoot].min_dev; }
    double max_over_occupied() const { return nodes_[kRoot].max_dev; }

    // After a rebalance relocated [slot_lo, slot_hi): recompute those slots' deviations from
    // current ranks. A grow/shrink reports the whole array and changes capacity, which we detect
    // and service as a full rebuild.
    void rebuild_range(std::size_t slot_lo, std::size_t slot_hi,
                       const PmaBlock& pma, LinearModel model, Key key_low) {
        if (resize_if_needed(pma.capacity())) {
            full_rebuild(pma, model, key_low);
            return;
        }

        if (slot_hi > leaf_count_) slot_hi = leaf_count_;
        if (slot_lo >= slot_hi) return;

        rebuild_span(kRoot, 0, leaf_count_, slot_lo, slot_hi, pma, model, key_low);
    }

    // After a refit swapped the model: every deviation changed.
    void rebuild_all(const PmaBlock& pma, LinearModel model, Key key_low) {
        resize_if_needed(pma.capacity());
        full_rebuild(pma, model, key_low);
    }

    std::size_t capacity() const { return leaf_count_; }

private:
    static constexpr std::size_t kRoot = 1;
    static constexpr double kEmptyMin = std::numeric_limits<double>::infinity();
    static constexpr double kEmptyMax = -std::numeric_limits<double>::infinity();

    struct Node {
        double min_dev = kEmptyMin;
        double max_dev = kEmptyMax;

        // lazy is just a deferred update, when walking down the node, then update its children, etc.
        // dont want to waste work
        double lazy = 0.0;

        // does the subtree under the current node contain at least one key?
        bool subtree_has_key = false;
    };

    std::vector<Node> nodes_;
    std::size_t leaf_count_ = 0;

    static std::size_t child_left(std::size_t node) { return 2 * node; }
    static std::size_t child_right(std::size_t node) { return 2 * node + 1; }

    // Shift a node's occupied contribution by delta. The guard is the whole trick: an empty
    // subtree owns no key to shift, so it takes neither the min/max change nor the lazy tag,
    // which is what keeps a later place() into one of its gaps from inheriting the shift.
    void apply_lazy(std::size_t node, double delta) {
        if (!nodes_[node].subtree_has_key) return;

        nodes_[node].min_dev += delta;
        nodes_[node].max_dev += delta;
        nodes_[node].lazy += delta;
    }

    void push_down(std::size_t node) {
        double pending = nodes_[node].lazy;
        if (pending == 0.0) return;

        apply_lazy(child_left(node), pending);
        apply_lazy(child_right(node), pending);
        nodes_[node].lazy = 0.0;
    }

    void pull_up(std::size_t node) {
        const Node& left = nodes_[child_left(node)];
        const Node& right = nodes_[child_right(node)];

        nodes_[node].subtree_has_key = left.subtree_has_key || right.subtree_has_key;
        nodes_[node].min_dev = std::min(left.min_dev, right.min_dev);
        nodes_[node].max_dev = std::max(left.max_dev, right.max_dev);
    }

    void set_leaf(std::size_t node, bool occupied, double deviation) {
        if (occupied) {
            nodes_[node].min_dev = deviation;
            nodes_[node].max_dev = deviation;
            nodes_[node].subtree_has_key = true;
        } else {
            nodes_[node].min_dev = kEmptyMin;
            nodes_[node].max_dev = kEmptyMax;
            nodes_[node].subtree_has_key = false;
        }

        nodes_[node].lazy = 0.0;
    }

    void point_update(std::size_t node, 
                      std::size_t node_lo, 
                      std::size_t node_hi,
                      std::size_t slot, 
                      bool occupied, 
                      double deviation) {
        if (node_hi - node_lo == 1) {
            set_leaf(node, occupied, deviation);
            return;
        }
        
        // notice this isnt recursive - peak laziness
        push_down(node);

        std::size_t mid = node_lo + (node_hi - node_lo) / 2;

        if (slot < mid) point_update(child_left(node), node_lo, mid, slot, occupied, deviation);
        else point_update(child_right(node), mid, node_hi, slot, occupied, deviation);

        pull_up(node);
    }

    void range_add(std::size_t node, 
                   std::size_t node_lo, 
                   std::size_t node_hi,
                   std::size_t lo, 
                   std::size_t hi, 
                   double delta) {
        if (hi <= node_lo || node_hi <= lo) return;

        if (lo <= node_lo && node_hi <= hi) {
            apply_lazy(node, delta);
            return;
        }

        push_down(node);

        std::size_t mid = node_lo + (node_hi - node_lo) / 2;

        range_add(child_left(node), node_lo, mid, lo, hi, delta);
        range_add(child_right(node), mid, node_hi, lo, hi, delta);

        pull_up(node);
    }

    // sets every leaf in [slot_lo, slot_hi) from current pma state
    // occupied slots in slot order carry consecutive ranks, 
    // so one Fenwick query at the low end plus a running counter gives
    // every rank without a per slot descent
    // TODO(perf): could jump occupied to occupied and bulk clear the gap runs
    // instead of one set leaf per empty slot (minor)
    void write_leaves(std::size_t slot_lo, 
                      std::size_t slot_hi,
                      const PmaBlock& pma, 
                      LinearModel model, 
                      Key key_low) {
        Rank running_rank = pma.local_rank(slot_lo);
        std::size_t next_occupied = pma.next_occupied(slot_lo);

        for (std::size_t slot = slot_lo; slot < slot_hi; ++slot) {
            std::size_t leaf = leaf_count_ + slot;

            if (slot == next_occupied) {
                double deviation = line_at(model, pma.key_at(slot), key_low)
                                   - static_cast<double>(running_rank);
                set_leaf(leaf, true, deviation);
                ++running_rank;
                next_occupied = pma.next_occupied(slot + 1);
            } else {
                set_leaf(leaf, false, 0.0);
            }
        }
    }

    // TODO(perf):
    // O(cap) called on every refit from rebuild_all
    //idea: Window leaf
    // make each leaf cover a pma leaf window, holding that window's minmax deviation instead of one leaf per slot. 
    // leaf_count drops so the node array and such all shrink band memory
    void full_rebuild(const PmaBlock& pma, LinearModel model, Key key_low) {
        write_leaves(0, leaf_count_, pma, model, key_low);

        for (std::size_t node = leaf_count_ - 1; node != 0; --node) {
            nodes_[node].lazy = 0.0;
            pull_up(node);
        }
    }

    // bottom up reebuild the internal nodes inside a subtree
    // clears each node's lazy
    void rebuild_internal(std::size_t node, std::size_t node_lo, std::size_t node_hi) {
        if (node_hi - node_lo == 1) return;

        std::size_t mid = node_lo + (node_hi - node_lo) / 2;

        rebuild_internal(child_left(node), node_lo, mid);
        rebuild_internal(child_right(node), mid, node_hi);

        nodes_[node].lazy = 0.0;
        pull_up(node);
    }

    // A fully covered subtree: rewrite its leaves from the pma and rebuild it
    // Each node's lazy prop is thrown away
    void rebuild_subtree(std::size_t node, 
                         std::size_t node_lo, 
                         std::size_t node_hi,
                         const PmaBlock& pma, 
                         LinearModel model, 
                         Key key_low) {
        write_leaves(node_lo, node_hi, pma, model, key_low);
        rebuild_internal(node, node_lo, node_hi);
    }

    void rebuild_span(std::size_t node, 
                      std::size_t node_lo, 
                      std::size_t node_hi,
                      std::size_t lo, 
                      std::size_t hi,
                      const PmaBlock& pma, 
                      LinearModel model, 
                      Key key_low) {
        if (hi <= node_lo || node_hi <= lo) return;

        if (lo <= node_lo && node_hi <= hi) {
            rebuild_subtree(node, node_lo, node_hi, pma, model, key_low);
            return;
        }

        push_down(node);

        std::size_t mid = node_lo + (node_hi - node_lo) / 2;

        rebuild_span(child_left(node), node_lo, mid, lo, hi, pma, model, key_low);
        rebuild_span(child_right(node), mid, node_hi, lo, hi, pma, model, key_low);

        pull_up(node);
    }

    // TODO(perf):
    // assign reallocates + zero fills 2*cap nodes on every cap change.
    // under churn a segment straddling this density boundary can oscillate badly
    // can probably keep some type of reusable buffer
    // or add a cap hysteresis. this is a measure though, might not be worth
    bool resize_if_needed(std::size_t new_capacity) {
        if (new_capacity == leaf_count_) return false;

        leaf_count_ = new_capacity;
        nodes_.assign(2 * leaf_count_, Node{});
        return true;
    }
};

}