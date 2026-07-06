/*
* pma.hpp - The Packed Memory Array
* 
* A mutable, pointer free, order storage: keys sit in sorted order in a gapped array (like ALEX)
* so an insert drops into a nearby gap and rebalances. One PmaBlock backs one eps seg
* 
* O(log^2(w)) amortized insert and delete.
* Amortized O(1) random insert
* O(1 + k) ordered scan
*
* Refefrences:
* "A Sparse Table Implementation of Priority Queues": https://dl.acm.org/doi/10.5555/646235.682700?__cf_chl_f_tk=1_HCYxIkOXiIp7PPeSkK0BhBoNl1RrmWCq2MtxB7qEo-1783354092-1.0.1.1-iQ8eWfUBnfOPa10g9nNUyqBNnw1POodoscahxEHr1_g
* "An Adaptive Packed memory Array": https://dl.acm.org/doi/10.1145/1292609.1292616
* Occupancy bitmap follows ALEX: https://dl.acm.org/doi/10.1145/3318464.3389711
*/

#pragma once

#include "status.hpp"
#include "types.hpp"
#include <vector>
#include <span>
#include <bit>
#include <cmath>
#include <cstdint>
#include <optional>

namespace li::detail {

template <class Payload = Rank>
class PmaBlock {
public:
    static PmaBlock bulk_load(std::span<const Key> keys, std::span<const Payload> payloads) {
        LI_ASSERT(keys.size() == payloads.size());
        PmaBlock b;
        std::size_t n = keys.size();
        b.capacity_ = choose_capacity(n);
        b.keys_.assign(b.capacity_, Key{});
        b.payloads_.assign(b.capacity_, Payload{});
        b.bits_.assign((b.capacity_ + 63) >> 6, 0);
        b.count_ = n;
        b.spread(0, b.capacity_, keys, payloads);
        return b;
    }

    std::size_t size() const { return count_; }
    std::size_t capacity() const { return capacity_; }
    bool empty() const { return count_ == 0; }

    Rank local_rank(std::size_t slot) const {
        if (slot >= capacity_) return count_;
        std::size_t w = slot >> 6, b = slot & 63, r = 0;
        for (std::size_t i = 0; i < w; ++i) r += std::popcount(bits_[i]);
        r += std::popcount(bits_[w] & ((b == 0) ? 0ull : ((1ull << b) - 1)));
        return r;
    }

    std::size_t slot_of_rank(Rank r) const {
        if (r >= count_) return capacity_;
        std::size_t seen = 0;
        for (std::size_t i = 0; i < num_words(); ++i) {
            std::size_t c = std::popcount(bits_[i]);
            if (seen + c > r) return (i << 6) + select_in_word(bits_[i], r - seen);
            seen += c;
        }
        return capacity_;
    }

    Key key_at(std::size_t slot) const { return keys_[slot]; }
    Payload payload_at(std::size_t slot) const { return payloads_[slot]; }

    std::size_t next_occupied(std::size_t from) const {
        std::size_t s = from;
        while (s < capacity_ && !is_occupied(s)) ++s;
        return s;
    }

    std::size_t lower_bound(Key key) const {
        std::size_t lo = 0, hi = capacity_;
        while (lo < hi) {
            std::size_t mid = lo + (hi - lo) / 2;
            std::size_t s = next_occupied(mid);
            if (s >= hi) hi = mid;
            else if (keys_[s] < key) lo = s + 1;
            else hi = s;
        }
        return next_occupied(lo);
    }

    std::optional<std::size_t> find(Key key) const {
        std::size_t s = lower_bound(key);
        if (s < capacity_ && keys_[s] == key) return s;
        return std::nullopt;
    }

    void insert(Key key, Payload payload) {
        std::size_t pos = lower_bound(key);
        LI_ASSERT(pos == capacity_ || keys_[pos] != key);
        std::size_t h = height();
        std::size_t lw = (pos == capacity_) ? (num_windows() - 1) : (pos / kLeafWindow);
        for (std::size_t level = 0; level <= h; ++level) {
            std::size_t base_leaf = (lw >> level) << level;
            std::size_t win_lo = base_leaf * kLeafWindow;
            std::size_t win_hi = win_lo + (kLeafWindow << level);
            double dens = double(occupied_in(win_lo, win_hi) + 1) / double(win_hi - win_lo);
            if (dens <= upper_threshold(level, h)) {
                rebalance_with(win_lo, win_hi, key, payload);
                ++count_;
                return;
            }
        }
        grow();
        insert(key, payload);
    }

    void append(Key key, Payload payload) {
        LI_ASSERT(count_ == 0 || key > keys_[last_occupied()]);
        if (count_ == 0) {
            place(0, key, payload);
            ++count_;
            return;
        }
        std::size_t pos = last_occupied() + 1;
        if (pos < capacity_) {
            std::size_t lw = pos / kLeafWindow;
            std::size_t wlo = lw * kLeafWindow;
            double dens = double(occupied_in(wlo, wlo + kLeafWindow) + 1) / double(kLeafWindow);
            if (dens <= upper_threshold(0, height())) {
                place(pos, key, payload);
                ++count_;
                return;
            }
        }
        insert(key, payload);
    }

    bool erase(Key key) {
        std::optional<std::size_t> slot = find(key);
        if (!slot) return false;
        clear_bit(*slot);
        --count_;
        while (capacity_ > kLeafWindow &&
               double(count_) / double(capacity_) < lower_threshold(height(), height())) {
            shrink();
        }
        return true;
    }

    template <class F>
    void scan(Key lo, Key hi, F&& visit) const {
        for (std::size_t s = lower_bound(lo); s < capacity_; ++s) {
            if (!is_occupied(s)) continue;
            if (keys_[s] >= hi) break;
            visit(keys_[s], payloads_[s]);
        }
    }

    void check_invariants() const {
        std::size_t c = 0;
        for (std::size_t i = 0; i < num_words(); ++i) c += std::popcount(bits_[i]);
        LI_ASSERT(c == count_);
        bool first = true;
        Key prev = 0;
        for (std::size_t s = 0; s < capacity_; ++s) {
            if (!is_occupied(s)) continue;
            if (!first) LI_ASSERT(keys_[s] > prev);
            prev = keys_[s];
            first = false;
        }
    }

private:
    std::vector<Key> keys_;
    std::vector<Payload> payloads_;
    std::vector<std::uint64_t> bits_;
    std::size_t count_ = 0;
    std::size_t capacity_ = 0;

    // TODO: same here, defaulting to 16 window slot. aybe do a some cost model to determine
    // at runtime, idk. 
    static constexpr std::size_t kLeafWindow = 16;
    static constexpr double kTauLeaf = 1.00, kTauRoot = 0.50;
    static constexpr double kRhoLeaf = 0.10, kRhoRoot = 0.20;

    // TODO: defaulting to .65 for now. same thing, cost model
    // based on storage requirements, etc. .65 is just a nice number :)
    static constexpr double kTargetDensity = 0.65;

    std::size_t num_words() const { return (capacity_ + 63) >> 6; }
    std::size_t num_windows() const { return capacity_ / kLeafWindow; }
    std::size_t height() const { return std::countr_zero(num_windows()); }

    bool is_occupied(std::size_t s) const { return (bits_[s >> 6] >> (s & 63)) & 1ull; }
    void set_bit(std::size_t s) { bits_[s >> 6] |= (1ull << (s & 63)); }
    void clear_bit(std::size_t s) { bits_[s >> 6] &= ~(1ull << (s & 63)); }
    void clear_bits(std::size_t lo, std::size_t hi) {
        for (std::size_t s = lo; s < hi; ++s) clear_bit(s);
    }

    void place(std::size_t slot, Key k, Payload p) {
        keys_[slot] = k;
        payloads_[slot] = p;
        set_bit(slot);
    }

    std::size_t first_occupied() const { return slot_of_rank(0); }
    std::size_t last_occupied() const {
        std::size_t s = capacity_;
        while (s > 0 && !is_occupied(s - 1)) --s;
        return s - 1;
    }

    std::size_t occupied_in(std::size_t lo, std::size_t hi) const {
        if (lo >= hi) return 0;
        std::size_t wlo = lo >> 6, whi = (hi - 1) >> 6;
        std::uint64_t lomask = ~0ull << (lo & 63);
        std::uint64_t himask = ~0ull >> (63 - ((hi - 1) & 63));
        if (wlo == whi) return std::popcount(bits_[wlo] & lomask & himask);
        std::size_t c = std::popcount(bits_[wlo] & lomask);
        for (std::size_t i = wlo + 1; i < whi; ++i) c += std::popcount(bits_[i]);
        c += std::popcount(bits_[whi] & himask);
        return c;
    }

    static double upper_threshold(std::size_t level, std::size_t h) {
        if (h == 0) return kTauLeaf;
        return kTauRoot + (kTauLeaf - kTauRoot) * double(h - level) / double(h);
    }
    static double lower_threshold(std::size_t level, std::size_t h) {
        if (h == 0) return kRhoLeaf;
        return kRhoRoot - (kRhoRoot - kRhoLeaf) * double(h - level) / double(h);
    }

    void spread(std::size_t lo, std::size_t hi,
                std::span<const Key> ks, std::span<const Payload> ps) {
        std::size_t n = ks.size(), slots = hi - lo;
        LI_ASSERT(n <= slots);
        clear_bits(lo, hi);
        for (std::size_t j = 0; j < n; ++j) {
            std::size_t s = lo + (j * slots) / n;
            place(s, ks[j], ps[j]);
        }
    }

    void rebalance_with(std::size_t lo, std::size_t hi, Key nk, Payload np) {
        std::vector<Key> ks;
        std::vector<Payload> ps;
        ks.reserve(occupied_in(lo, hi) + 1);
        ps.reserve(ks.capacity());
        bool inserted = false;
        for (std::size_t s = lo; s < hi; ++s) {
            if (!is_occupied(s)) continue;
            if (!inserted && keys_[s] > nk) {
                ks.push_back(nk);
                ps.push_back(np);
                inserted = true;
            }
            ks.push_back(keys_[s]);
            ps.push_back(payloads_[s]);
        }
        if (!inserted) {
            ks.push_back(nk);
            ps.push_back(np);
        }
        spread(lo, hi, ks, ps);
    }

    void resize_to(std::size_t new_cap) {
        std::vector<Key> ks;
        std::vector<Payload> ps;
        ks.reserve(count_);
        ps.reserve(count_);
        for (std::size_t s = 0; s < capacity_; ++s) {
            if (!is_occupied(s)) continue;
            ks.push_back(keys_[s]);
            ps.push_back(payloads_[s]);
        }
        keys_.assign(new_cap, Key{});
        payloads_.assign(new_cap, Payload{});
        capacity_ = new_cap;
        bits_.assign(num_words(), 0);
        spread(0, new_cap, ks, ps);
    }

    void grow() { resize_to(capacity_ * 2); }
    void shrink() { resize_to(capacity_ / 2); }

    static std::size_t select_in_word(std::uint64_t w, std::size_t k) {
        for (std::size_t i = 0; i < 64; ++i) {
            if (w & 1ull) {
                if (k == 0) return i;
                --k;
            }
            w >>= 1;
        }
        return 64;
    }

    static std::size_t choose_capacity(std::size_t n) {
        std::size_t need = (n == 0) ? 1 : std::size_t(std::ceil(double(n) / kTargetDensity));
        if (need < kLeafWindow) need = kLeafWindow;
        std::size_t windows = (need + kLeafWindow - 1) / kLeafWindow;
        return kLeafWindow * std::bit_ceil(windows);
    }
};

}