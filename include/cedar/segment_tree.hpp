/*
* segment_tree.hpp - The mapping table. A cache friendly B+tree with subtree counts in its nodes
*
* CEDAR's mapping table holds M ordered segment descriptors

* This is not a computed router, it is a dictionary with rank queries
* Tt must answer four things:
*   route(key): which segment owns the key
*   at(i): pointer to segment at index i
*   insert_at/erase_at: ordered updates on the tree
*
*
* Why:
* Why a segment tree and not a computed router like many other indexes?
* Most maintenance over other computed routers are amortised and not bounded well
* It is hard to bound the routing part of computation because of model rebuilding 
* and cascading effects
* Other indexes similar to mine (Gaede et al) use a balanced BST at the structural layer
*
*
* Worst Case Bounds:
* search/select/rank: O(log_B M)
* Insert/erase: a leaf split cascades at most to the root, so O(B * log_B M)
*
*
* Subtree Counts:
* Each inner slot caches sub_items (# segments in its child subtree),
* this gives rank in one descent.
*
*
* Not Copied, but based on:
* 
*   [Comer 1979] "The Ubiquitous B-Tree", https://dl.acm.org/doi/pdf/10.1145/356770.356776
*   [Rao & Ross 2000] "Making B+-trees Cache Conscious in Main Memory", https://dl.acm.org/doi/abs/10.1145/342009.335449
*   lx/STX B+tree: github.com/tlx/tlx
*
*
*/

#pragma once

#include "status.hpp"

#include <vector>
#include <memory>
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace li::detail {

// B is fanout
// slot = {Key, unique_ptr} = 16 bytes
// B=32, 32 * 16bytes = 512bytes / 64 = 8 cache lines
template <class Item, int B = 32>
class SegmentTree {
    static_assert(B >= 4 && (B % 2) == 0, "B must be even and >= 4");

public:
    SegmentTree() : root_(new Leaf()), height_(0), size_(0) {}
    ~SegmentTree() { destroy(root_, height_); }

    SegmentTree(const SegmentTree&) = delete;
    SegmentTree& operator=(const SegmentTree&) = delete;

    void clear() {
        destroy(root_, height_);
        root_ = new Leaf(); height_ = 0; size_ = 0;
    }

    std::size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }


    struct Located { 
        std::size_t index; 
        const Item* item; 
    };

    Located route_located(Key key) const {
        if (empty()) return { 0, nullptr };
        const Node* n = root_;
        std::size_t base = 0;

        for (int h = height_; h > 0; --h) {
            const Inner* in = static_cast<const Inner*>(n);
            int s = 0;

            while (s < in->n - 1 && key >= in->keys[s + 1]) {
                base += in->sub_items[s];
                ++s;
            }

            n = in->child[s];
        }
        
        const Leaf* lf = static_cast<const Leaf*>(n);
        int s = 0;
        while (s < lf->n - 1 && key >= lf->keys[s + 1]) ++s;
        return { base + std::size_t(s), &lf->items[s] };
    }

    // items i and i+1, the neighbor is the next leaf slot
    struct ItemPair { 
        Item* a; 
        Item* b; 
    };

    ItemPair at_pair(std::size_t i) {
        CEDAR_ASSERT(i + 1 < size_);

        Node* n = root_;
        std::size_t rem = i;

        for (int h = height_; h > 0; --h) {
            Inner* in = static_cast<Inner*>(n);
            int s = 0;

            while (s < in->n - 1 && rem >= in->sub_items[s]) {
                rem -= in->sub_items[s];
                ++s;
            }

            n = in->child[s];
        }

        Leaf* lf = static_cast<Leaf*>(n);
        Item* a = &lf->items[rem];
        Item* b = (int(rem) + 1 < lf->n) ? &lf->items[rem + 1] : &lf->next->items[0];

        return { a, b };
    }

    // the i'th item in key order
    Item& at(std::size_t i) {
        CEDAR_ASSERT(i < size_);
        Node* n = root_;

        for (int h = height_; h > 0; --h) {
            Inner* in = static_cast<Inner*>(n);
            int s = 0;

            while (s < in->n - 1 && i >= in->sub_items[s]) {
                i -= in->sub_items[s];
                ++s;
            }

            n = in->child[s];
        }

        return static_cast<Leaf*>(n)->items[i];
    }
    
    const Item& at(std::size_t i) const { 
        return const_cast<SegmentTree*>(this)->at(i); 
    }

    // returns index of segment that holds key
    std::size_t route(Key key) const {
        if (empty()) return 0;
        Node* n = root_;
        std::size_t base = 0;
        for (int h = height_; h > 0; --h) {
            const Inner* in = static_cast<const Inner*>(n);
            int s = 0;
            while (s < in->n - 1 && key >= in->keys[s + 1]) {
                base += in->sub_items[s];
                ++s;
            }
            n = in->child[s];
        }
        const Leaf* lf = static_cast<const Leaf*>(n);
        int s = 0;
        while (s < lf->n - 1 && key >= lf->keys[s + 1]) ++s;
        return base + std::size_t(s);
    }



    void insert_at(std::size_t i, Key sep, Item&& item) {
        CEDAR_ASSERT(i <= size_);

        Split sp = do_insert(root_, height_, i, sep, std::move(item));

        if (sp.node) {
            Inner* nr = new Inner();

            nr->n = 2;
            nr->keys[0] = first_key(root_, height_);
            nr->child[0] = root_;
            nr->keys[1] = sp.key;
            nr->child[1] = sp.node;
            nr->sub_items[0] = count_items(root_, height_);
            nr->sub_items[1] = count_items(sp.node, height_);

            root_ = nr;

            ++height_;
        }

        ++size_;
    }

    Item erase_at(std::size_t i) {
        CEDAR_ASSERT(i < size_);
        Item out{};
        do_erase(root_, height_, i, out);

        while (height_ > 0 && static_cast<Inner*>(root_)->n == 1) {

            Inner* old = static_cast<Inner*>(root_);
            root_ = old->child[0];
            old->n = 0;
            delete old;
            --height_;
        }

        --size_;
        return out;
    }

    // foreach scan from leftmost key. leaves have pointers to sibling 
    // leaf nodes so makes this easy linear scan once down there
    template <class F>
    void for_each_item(F&& f) const {
        const Node* n = root_;

        for (int h = height_; h > 0; --h) {
            n = static_cast<const Inner*>(n)->child[0];
        }

        for (const Leaf* lf = static_cast<const Leaf*>(n); lf; lf = lf->next) {
            
            for (int k = 0; k < lf->n; ++k) {
                f(lf->items[k]);
            }
        }
    }

    // **IMPORTANT** 
    // the seg tree routes on copies of key_low, so anything that changes a seg's key_low needs
    // to call this or the oruting desyncs
    void set_sep(std::size_t i, Key sep) {
        CEDAR_ASSERT(i < size_);
        Node* n = root_;
        std::size_t rem = i;

        for (int h = height_; h > 0; --h) {
            Inner* in = static_cast<Inner*>(n);
            int s = 0;

            while (s < in->n - 1 && rem >= in->sub_items[s]) {
                rem -= in->sub_items[s];
                ++s;
            }

            if (rem == 0) in->keys[s] = sep;
            n = in->child[s];
        }

        static_cast<Leaf*>(n)->keys[rem] = sep;
    }

    void check_invariants() const {
        if (empty()) {
            CEDAR_ASSERT(height_ == 0 && root_->n == 0);
            return;
        }

        verify(root_, height_, true);
    }


private:
    struct Node {
        int n = 0; 
    };

    struct Leaf : Node {
        Key keys[B];
        Item items[B];
        Leaf* next = nullptr;
        // only forward linear scans right now
        // Leaf* prev = nullptr;
    };

    struct Inner : Node {
        // keys[i] == child_i's lowest key
        Key keys[B];
        Node* child[B];
        std::uint64_t sub_items[B];
    };

    struct Split { 
        Node* node = nullptr; 
        Key key = 0; 
    };

    Node* root_;
    int height_;
    std::size_t size_;

    static Key first_key(Node* n, int h) {
        return h == 0 ? static_cast<Leaf*>(n)->keys[0]
                      : static_cast<Inner*>(n)->keys[0];
    }


    static std::uint64_t count_items(Node* n, int h) {
        if (h == 0) {
            return std::uint64_t(n->n);
        }

        Inner* in = static_cast<Inner*>(n);
        std::uint64_t s = 0;

        for (int i = 0; i < in->n; ++i) {
            s += in->sub_items[i];
        }

        return s;
    }

    static void destroy(Node* n, int h) {
        if (h > 0) {
            Inner* in = static_cast<Inner*>(n);
            for (int i = 0; i < in->n; ++i) destroy(in->child[i], h - 1);

            delete in;
        } else {
            delete static_cast<Leaf*>(n);
        }
    }

    Split do_insert(Node* n, int h, std::size_t i, Key sep, Item&& item) {
        if (h == 0) {
            Leaf* lf = static_cast<Leaf*>(n);

            if (lf->n < B) {
                leaf_place(lf, int(i), sep, std::move(item));
                return {};
            }

            Leaf* rt = new Leaf();
            const int mid = B / 2;

            for (int k = mid; k < B; ++k) {
                rt->keys[k - mid] = lf->keys[k];
                rt->items[k - mid] = std::move(lf->items[k]);
            }

            rt->n = B - mid; lf->n = mid;
            rt->next = lf->next; 
            // rt->prev = lf;

            // if (lf->next) {
            //     lf->next->prev = rt;
            // }

            lf->next = rt;

            if (int(i) <= mid) {
                leaf_place(lf, int(i), sep, std::move(item));
            } else {
                leaf_place(rt, int(i) - mid, sep, std::move(item));
            }

            return { rt, rt->keys[0] };
        }

        Inner* in = static_cast<Inner*>(n);
        int s = 0;
        std::size_t rem = i;
        
        while (s < in->n - 1 && rem > in->sub_items[s]) {
            rem -= in->sub_items[s];
            ++s;
        }

        Split sp = do_insert(in->child[s], h - 1, rem, sep, std::move(item));
        in->sub_items[s] = count_items(in->child[s], h - 1);
        in->keys[s] = first_key(in->child[s], h - 1);

        if (!sp.node) return {};

        if (in->n < B) {
            inner_place(in, s + 1, sp.key, sp.node, h - 1);
            return {};
        }

        Inner* rt = new Inner();
        const int mid = B / 2;

        for (int k = mid; k < B; ++k) {
            rt->keys[k - mid] = in->keys[k];
            rt->child[k - mid] = in->child[k];
            rt->sub_items[k - mid] = in->sub_items[k];
        }

        rt->n = B - mid; in->n = mid;

        if (s + 1 <= mid) inner_place(in, s + 1, sp.key, sp.node, h - 1);
        else inner_place(rt, s + 1 - mid, sp.key, sp.node, h - 1);

        return { rt, rt->keys[0] };
    }

    static void leaf_place(Leaf* lf, int at, Key sep, Item&& item) {
        for (int k = lf->n; k > at; --k) {
            lf->keys[k] = lf->keys[k - 1];
            lf->items[k] = std::move(lf->items[k - 1]);
        }
        lf->keys[at] = sep;
        lf->items[at] = std::move(item);
        ++lf->n;
    }

    static void inner_place(Inner* in, int at, Key k, Node* c, int ch) {
        for (int j = in->n; j > at; --j) {
            in->keys[j] = in->keys[j - 1];
            in->child[j] = in->child[j - 1];
            in->sub_items[j] = in->sub_items[j - 1];
        }

        in->keys[at] = k;
        in->child[at] = c;
        in->sub_items[at] = count_items(c, ch);
        ++in->n;
    }

    void do_erase(Node* n, int h, std::size_t i, Item& out) {
        if (h == 0) {
            Leaf* lf = static_cast<Leaf*>(n);
            out = std::move(lf->items[i]);

            for (int k = int(i); k + 1 < lf->n; ++k) {
                lf->keys[k] = lf->keys[k + 1];
                lf->items[k] = std::move(lf->items[k + 1]);
            }

            --lf->n;
            return;
        }

        Inner* in = static_cast<Inner*>(n);
        int s = 0;
        std::size_t rem = i;

        while (s < in->n - 1 && rem >= in->sub_items[s]) {
            rem -= in->sub_items[s];
            ++s;
        }

        do_erase(in->child[s], h - 1, rem, out);
        rebalance_child(in, s, h);

        // only children s-1 to s+1 can have changed
        const int lo = (s > 0) ? s - 1 : 0;
        const int hi = std::min(in->n, s + 2);
        for (int k = lo; k < hi; ++k) {
            in->sub_items[k] = count_items(in->child[k], h - 1);
            in->keys[k] = first_key(in->child[k], h - 1);
        }
    }

    void rebalance_child(Inner* in, int s, int h) {
        const int minn = B / 2;
        Node* c = in->child[s];

        if (c->n >= minn) return;

        if (h - 1 == 0) {
            Leaf* cl = static_cast<Leaf*>(c);
            if (s > 0 && static_cast<Leaf*>(in->child[s - 1])->n > minn) {
                Leaf* L = static_cast<Leaf*>(in->child[s - 1]);
                leaf_place(cl, 0, L->keys[L->n - 1], std::move(L->items[L->n - 1]));
                --L->n; return;
            }

            if (s + 1 < in->n && static_cast<Leaf*>(in->child[s + 1])->n > minn) {
                Leaf* R = static_cast<Leaf*>(in->child[s + 1]);
                leaf_place(cl, cl->n, R->keys[0], std::move(R->items[0]));
                for (int k = 0; k + 1 < R->n; ++k) {
                    R->keys[k] = R->keys[k + 1];
                    R->items[k] = std::move(R->items[k + 1]);
                }
                --R->n; return;
            }

            int a = (s > 0) ? s - 1 : s;
            if (a + 1 >= in->n) return;

            Leaf* L = static_cast<Leaf*>(in->child[a]);
            Leaf* R = static_cast<Leaf*>(in->child[a + 1]);

            for (int k = 0; k < R->n; ++k) {
                L->keys[L->n + k] = R->keys[k];
                L->items[L->n + k] = std::move(R->items[k]);
            }

            L->n += R->n;
            L->next = R->next; 
            // if (R->next) R->next->prev = L;

            delete R;
            inner_remove(in, a + 1);
        } else {
            Inner* cc = static_cast<Inner*>(c);

            if (s > 0 && in->child[s - 1]->n > minn) {
                Inner* L = static_cast<Inner*>(in->child[s - 1]);
                inner_place(cc, 0, L->keys[L->n - 1], L->child[L->n - 1], h - 2);
                --L->n; return;
            }

            if (s + 1 < in->n && in->child[s + 1]->n > minn) {
                Inner* R = static_cast<Inner*>(in->child[s + 1]);
                inner_place(cc, cc->n, R->keys[0], R->child[0], h - 2);
                for (int k = 0; k + 1 < R->n; ++k) { R->keys[k] = R->keys[k+1]; R->child[k] = R->child[k+1];
                    R->sub_items[k] = R->sub_items[k+1]; }
                --R->n; return;
            }

            int a = (s > 0) ? s - 1 : s;
            if (a + 1 >= in->n) return;

            Inner* L = static_cast<Inner*>(in->child[a]);
            Inner* R = static_cast<Inner*>(in->child[a + 1]);

            for (int k = 0; k < R->n; ++k) {
                inner_place(L, L->n, R->keys[k], R->child[k], h - 2);
            }

            R->n = 0; delete R;
            inner_remove(in, a + 1);
        }
    }

    static void inner_remove(Inner* in, int at) {
        for (int k = at; k + 1 < in->n; ++k) {
            in->keys[k] = in->keys[k + 1];
            in->child[k] = in->child[k + 1];
            in->sub_items[k] = in->sub_items[k + 1];
        }

        --in->n;
    }

    void verify(Node* n, int h, bool is_root) const {
        CEDAR_ASSERT(n->n >= 1);

        if (!is_root) CEDAR_ASSERT(n->n >= B / 2);

        CEDAR_ASSERT(n->n <= B);
        
        if (h == 0) return;

        const Inner* in = static_cast<const Inner*>(n);

        for (int i = 0; i < in->n; ++i) {
            CEDAR_ASSERT(in->sub_items[i] == count_items(in->child[i], h - 1));
            CEDAR_ASSERT(in->keys[i] == first_key(in->child[i], h - 1));
            if (i) CEDAR_ASSERT(in->keys[i - 1] <= in->keys[i]);
            verify(in->child[i], h - 1, false);
        }
    }
};

}