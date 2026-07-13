#pragma once

#include "types.hpp"
#include "status.hpp"

#include <vector>
#include <span>
#include <cstdint>
#include <algorithm>
#include <utility>
#include "model.hpp"

namespace li::detail {

// DEPRACATED
// not used anymore, this is algorithmically optimal,
// but horribly cache unfriendly - so moving on from hull trees and tatic hull
// see here: https://drops.dagstuhl.de/storage/00lipics/lipics-vol351-esa2025/LIPIcs.ESA.2025.64/LIPIcs.ESA.2025.64.pdf
class HullTree {
public:
    static constexpr std::uint32_t NIL = 0xFFFFFFFFu;

    HullTree() = default;

    static HullTree bulk_build(std::span<const Key> keys) {
        HullTree t;
        if (keys.empty()) return t;

        t.moment_origin_ = keys.front();
        t.items_.assign(keys.begin(), keys.end());
        t.root_ = t.build_balanced(0, t.items_.size());

        return t;
    }

    bool empty() const { 
        return root_ == NIL; 
    }

    std::size_t size() const { 
        return root_ == NIL ? 0 : nodes_[root_].size; 
    }

    bool contains(Key key) const {
        std::uint32_t cur = root_;

        while (cur != NIL && !is_leaf(cur)) {
            cur = (key < nodes_[nodes_[cur].right].min_key) ? nodes_[cur].left : nodes_[cur].right;
        }

        return cur != NIL && nodes_[cur].min_key == key;
    }

    // y = rank 
    Rank rank_of(Key key) const {
        if (root_ == NIL) return 0;
        Rank r = 0;

        std::uint32_t cur = root_;
        while (!is_leaf(cur)) {
            if (key < nodes_[nodes_[cur].right].min_key) {
                cur = nodes_[cur].left;
            } else {
                r += nodes_[nodes_[cur].left].size;
                cur = nodes_[cur].right;
            }
        }

        if (nodes_[cur].min_key < key) r += 1;
        return r;
    }
    
    // this is the inverse of rank_of()
    // Rank -> Key
    Key select(Rank r) const {
        LI_ASSERT(r < size());
        std::uint32_t cur = root_;

        while (!is_leaf(cur)) {
            const Rank ls = nodes_[nodes_[cur].left].size;

            if (r < ls) { 
                cur = nodes_[cur].left; 
            } else { 
                r -= ls; 
                cur = nodes_[cur].right; 
            }
        }

        return nodes_[cur].min_key;
    }

    void insert(Key key) {
        if (root_ == NIL) {
            moment_origin_ = key;
            root_ = new_leaf(key);
            return;
        }

        path_.clear();
        std::uint32_t cur = root_;
        while (!is_leaf(cur)) {
            path_.push_back(cur);
            cur = (key < nodes_[nodes_[cur].right].min_key) ? nodes_[cur].left : nodes_[cur].right;
        }

        LI_ASSERT(nodes_[cur].min_key != key);

        const std::uint32_t leaf_idx = cur;
        const std::uint32_t added = new_leaf(key);
        const std::uint32_t inner = (key < nodes_[leaf_idx].min_key) 
                                    ? new_internal(added, leaf_idx) 
                                    : new_internal(leaf_idx, added);

        if (path_.empty()) {
            root_ = inner;
        } else {
            const std::uint32_t parent = path_.back();

            if (nodes_[parent].left == leaf_idx) {
                nodes_[parent].left = inner;
            } else {
                nodes_[parent].right = inner;
            }
        }

        fix_up();
        rebalance();
    }

    bool erase(Key key) {
        if (root_ == NIL) return false;

        path_.clear();
        std::uint32_t cur = root_;
        while (!is_leaf(cur)) {
            path_.push_back(cur);
            cur = (key < nodes_[nodes_[cur].right].min_key) ? nodes_[cur].left : nodes_[cur].right;
        }
        if (nodes_[cur].min_key != key) return false;

        const std::uint32_t leaf_idx = cur;
        if (path_.empty()) {
            free_node(leaf_idx);
            root_ = NIL;
            return true;
        }

        const std::uint32_t parent = path_.back();
        const std::uint32_t sibling = (nodes_[parent].left == leaf_idx)
                                    ? nodes_[parent].right : nodes_[parent].left;

        if (path_.size() == 1) {
            root_ = sibling;
        } else {
            const std::uint32_t grandparent = path_[path_.size() - 2];
            if (nodes_[grandparent].left == parent) nodes_[grandparent].left = sibling;
            else nodes_[grandparent].right = sibling;
        }

        free_node(leaf_idx);
        free_node(parent);

        path_.pop_back();
        fix_up();
        rebalance();
        return true;
    }

    template <class F>
    void for_each(F&& visit) const {
        if (root_ != NIL) in_order(root_, visit);
    }

    void validate() const {
        if (root_ == NIL) return;

        Key prev{};
        bool seen = false;

        validate_node(root_, prev, seen);
    }

    LinearModel model(Key key_low) const {
        LI_ASSERT(root_ != NIL);

        LinearModel line = least_squares(nodes_[root_].moments);

        const double key_low_offset =
            static_cast<double>(key_low) - static_cast<double>(moment_origin_);
        line.beta = line.beta + line.alpha * key_low_offset;

        return line;
    }

    const LeastSquaresSums& root_moments_for_test() const {
        LI_ASSERT(root_ != NIL);
        return nodes_[root_].moments;
    }

    Key moment_origin_for_test() const {
        return moment_origin_;
    }


private:
    struct Node {
        Key           min_key;
        std::uint32_t left;
        std::uint32_t right;
        std::uint32_t size;
        LeastSquaresSums moments;
    };

    // why 0.71?
    // https://en.wikipedia.org/wiki/Weight-balanced_tree
    // Nievergelt and Reingold weight balance threshold
    // alpha < 1 - (sqrt2/2). sqrt2/2 ~ .71
    static constexpr double kMaxChildFraction = 0.71;

    
    // TODO:
    //  1.  grid search this paremeter
    //  2. hybrid build: only segments with w >= kHullTreeMinWidth carry a tree
    //  3. brodal jacob fully dynamic hull is O(logw) vs this one's O(log^2w).. keep in mind

    std::vector<Node> nodes_;
    std::vector<std::uint32_t> free_;
    // path_ is a reusable buffer to track the path so we know what needs rebalancing after an insert/delete
    std::vector<std::uint32_t> path_;
    std::vector<Key> items_;
    std::uint32_t root_ = NIL;
    Key moment_origin_ = 0;

    bool is_leaf(std::uint32_t n) const { 
        return nodes_[n].left == NIL; 
    }

    // arena allocation for speeeed >:D
    // love arenas
    std::uint32_t alloc() {
        if (!free_.empty()) { 
            std::uint32_t i = free_.back(); 
            free_.pop_back(); 
            return i; 
        }

        nodes_.push_back(Node{});
        return static_cast<std::uint32_t>(nodes_.size() - 1);
    }

    void free_node(std::uint32_t i) { 
        free_.push_back(i); 
    }

    std::uint32_t new_leaf(Key key) {
        std::uint32_t index = alloc();
        Node& node = nodes_[index];

        node.min_key = key;
        node.left = NIL;
        node.right = NIL;
        node.size = 1;

        const double x = static_cast<double>(key) - static_cast<double>(moment_origin_);

        node.moments = LeastSquaresSums{};
        node.moments.add(x, 0.0);

        return index;
    }

    std::uint32_t new_internal(std::uint32_t l, std::uint32_t r) {
        std::uint32_t i = alloc();

        nodes_[i].left = l;
        nodes_[i].right = r;
        nodes_[i].size = nodes_[l].size + nodes_[r].size;
        nodes_[i].min_key = nodes_[l].min_key;
        nodes_[i].moments = combine_child_moments(nodes_[l], nodes_[r]);

        return i;
    }

    void refresh(std::uint32_t n) {
        const std::uint32_t left = nodes_[n].left;
        const std::uint32_t right = nodes_[n].right;

        nodes_[n].size = nodes_[left].size + nodes_[right].size;
        nodes_[n].min_key = nodes_[left].min_key;

        nodes_[n].moments = combine_child_moments(nodes_[left], nodes_[right]);
    }

    void fix_up() {
        for (std::size_t i = path_.size(); i-- > 0; ) {
            refresh(path_[i]);
        }
    }

    bool unbalanced(std::uint32_t n) const {
        const std::uint32_t bigger =
            std::max(nodes_[nodes_[n].left].size, nodes_[nodes_[n].right].size);
        return static_cast<double>(bigger) > kMaxChildFraction * static_cast<double>(nodes_[n].size);
    }

    void rebalance() {
        for (std::size_t i = 0; i < path_.size(); ++i) {
            if (unbalanced(path_[i])) {
                const std::uint32_t parent = (i == 0) ? NIL : path_[i - 1];
                rebuild(path_[i], parent);
                return;
            }
        }
    }

    void rebuild(std::uint32_t sub, std::uint32_t parent) {
        const bool is_left = (parent != NIL) && (nodes_[parent].left == sub);
        items_.clear();
        
        collect(sub, items_);
        gather_free(sub);
        
        const std::uint32_t nr = build_balanced(0, items_.size());
        
        if (parent == NIL) root_ = nr;
        else if (is_left) nodes_[parent].left = nr;
        else nodes_[parent].right = nr;
    }

    void collect(std::uint32_t n, std::vector<Key>& out) const {
        if (is_leaf(n)) { out.push_back(nodes_[n].min_key); return; }
        
        collect(nodes_[n].left, out);
        collect(nodes_[n].right, out);
    }
    void gather_free(std::uint32_t n) {
        if (!is_leaf(n)) { gather_free(nodes_[n].left); gather_free(nodes_[n].right); }
        free_node(n);
    }

    std::uint32_t build_balanced(std::size_t lo, std::size_t hi) {
        if (hi - lo == 1) return new_leaf(items_[lo]);
        
        const std::size_t mid = lo + (hi - lo) / 2;
        const std::uint32_t l = build_balanced(lo, mid);
        const std::uint32_t r = build_balanced(mid, hi);
        
        return new_internal(l, r);
    }

    template <class F>
    void in_order(std::uint32_t n, F& visit) const {
        if (is_leaf(n)) { visit(nodes_[n].min_key); return; }
        in_order(nodes_[n].left, visit);
        in_order(nodes_[n].right, visit);
    }

    void validate_node(std::uint32_t n, Key& prev, bool& seen) const {
        LI_ASSERT(static_cast<std::uint64_t>(nodes_[n].size) == nodes_[n].moments.n);

        if (is_leaf(n)) {
            if (seen) LI_ASSERT(nodes_[n].min_key > prev);
            prev = nodes_[n].min_key;
            seen = true;
            LI_ASSERT(nodes_[n].size == 1);
            return;
        }

        LI_ASSERT(nodes_[n].left != NIL && nodes_[n].right != NIL);
        validate_node(nodes_[n].left, prev, seen);
        validate_node(nodes_[n].right, prev, seen);

        LI_ASSERT(nodes_[n].size == nodes_[nodes_[n].left].size + nodes_[nodes_[n].right].size);
        LI_ASSERT(nodes_[n].min_key == nodes_[nodes_[n].left].min_key);
        LI_ASSERT(!unbalanced(n));
    }

    
    static LeastSquaresSums combine_child_moments(const Node& left, const Node& right) {
        const double left_count = static_cast<double>(left.size);

        LeastSquaresSums parent_moments;
        parent_moments.n      = left.moments.n + right.moments.n;
        parent_moments.sum_x  = left.moments.sum_x + right.moments.sum_x;
        parent_moments.sum_xx = left.moments.sum_xx + right.moments.sum_xx;

        // y = rank. right child numbered its points from 0 locally, but they actually
        // sit after all of lefts points, so every right rank is left_count higher
        // just ad the shif tback
        parent_moments.sum_y  = left.moments.sum_y + right.moments.sum_y
                              + left_count * static_cast<double>(right.moments.n);
        parent_moments.sum_xy = left.moments.sum_xy + right.moments.sum_xy
                              + left_count * right.moments.sum_x;

        return parent_moments;
    }
};

}