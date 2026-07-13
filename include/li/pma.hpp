/*
* pma.hpp - The Packed Memory Array
*
* A mutable, pointer free, ordered store: keys sit in sorted order in a gapped array so an insert
* drops into a nearby gap and rebalances. One PmaBlock backs one epsilon-segment.
*
* The block never grows past a ceiling derived from the key cap W_s. The caller enforces the KEY
* cap (cap-split when count > W_s); this file enforces the SLOT ceiling so any single rebalance or
* resize is O(W_s) worst case and there is no unbounded doubling.
*
* O(W_s) worst case insert/delete (bounded by the cap); amortized cheap; O(1) trailing append.
*
* References:
* "A Sparse Table Implementation of Priority Queues": https://dl.acm.org/doi/10.5555/646235.682700
* "An Adaptive Packed Memory Array": https://dl.acm.org/doi/10.1145/1292609.1292616
* Occupancy bitmap follows ALEX: https://dl.acm.org/doi/10.1145/3318464.3389711
*/

#pragma once

#include "status.hpp"
#include "types.hpp"

#include <vector>
#include <span>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace li::detail {

class PmaBlock {
public:
    // moved_lo == moved_hi means nothing was relocated (the band only needs the local place/clear
    // plus the rank shift). Otherwise [moved_lo, moved_hi) is the slot range whose occupants moved
    // and must be rebuilt in the band.
    struct EditResult {
        std::size_t slot;
        std::size_t moved_lo;
        std::size_t moved_hi;
        bool found;

        bool moved_empty() const { return moved_lo == moved_hi; }
    };


    static std::size_t capacity_for_key_cap(std::size_t key_cap) {
        double needed_slots = std::ceil(double(key_cap + 1) / kMaxDensityRoot);

        std::size_t needed = static_cast<std::size_t>(needed_slots);
        if (needed < kLeafWindow) needed = kLeafWindow;

        std::size_t windows = (needed + kLeafWindow - 1) / kLeafWindow;

        return kLeafWindow * std::bit_ceil(windows);
    }

    static PmaBlock bulk_load(std::span<const Key> keys, std::span<const Payload> payloads,
                              std::size_t max_capacity_slots) {
        LI_ASSERT(keys.size() == payloads.size());

        PmaBlock block;
        std::size_t count = keys.size();

        block.max_capacity_ = max_capacity_slots;
        block.capacity_ = choose_capacity(count, max_capacity_slots);

        block.keys_.assign(block.capacity_, Key{});
        block.payloads_.assign(block.capacity_, Payload{});
        block.bits_.assign((block.capacity_ + 63) >> 6, 0);
        block.fen_.assign(block.num_words() + 1, 0);
        block.count_ = count;

        block.spread(0, block.capacity_, keys, payloads);

        return block;
    }

    std::size_t size() const { return count_; }
    std::size_t capacity() const { return capacity_; }
    std::size_t max_capacity() const { return max_capacity_; }
    bool empty() const { return count_ == 0; }

    // TODO(perf): cache the max key as a member, O(1) 
    // last_occupied is O(log C + 64) right now
    Key max_key() const {
        LI_ASSERT(count_ > 0);
        return keys_[last_occupied()];
    }

    Rank local_rank(std::size_t slot) const {
        if (slot >= capacity_) return count_;

        std::size_t word = slot >> 6;
        std::size_t bit = slot & 63;

        Rank rank = fen_prefix(word);
        rank += std::popcount(bits_[word] & ((bit == 0) ? 0ull : ((1ull << bit) - 1)));

        return rank;
    }

    // TODO(perf): 
    // select_in_word is a 64 count loop
    // PDEP(1ull << k, word) + countr_zero (BMI2)
    std::size_t slot_of_rank(Rank rank) const {
        if (rank >= count_) return capacity_;

        std::size_t words = num_words();
        std::size_t pos = 0;
        Rank remaining = rank;

        for (std::size_t step = std::bit_floor(words); step; step >>= 1) {
            if (pos + step <= words && fen_[pos + step] <= remaining) {
                pos += step;
                remaining -= fen_[pos];
            }
        }

        return (pos << 6) + select_in_word(bits_[pos], remaining);
    }

    Key key_at(std::size_t slot) const { return keys_[slot]; }
    Payload payload_at(std::size_t slot) const { return payloads_[slot]; }

    // TODO(perf): 
    // scalar per slot scan, skip whole empty words via countr_zero(bits_[word] & mask)
    std::size_t next_occupied(std::size_t from) const {
        std::size_t slot = from;
        while (slot < capacity_ && !is_occupied(slot)) ++slot;
        return slot;
    }

    // TODO(perf): O(log C) serial cache misses
    // id prefer a model predicted slot + find_in over a bounded window
    // else a branchless or prefetched search
    std::size_t lower_bound(Key key) const {
        std::size_t lo = 0;
        std::size_t hi = capacity_;

        while (lo < hi) {
            std::size_t mid = lo + (hi - lo) / 2;
            std::size_t slot = next_occupied(mid);

            if (slot >= hi) hi = mid;
            else if (keys_[slot] < key) lo = slot + 1;
            else hi = slot;
        }

        return next_occupied(lo);
    }

    std::optional<std::size_t> find(Key key) const {
        std::size_t slot = lower_bound(key);
        if (slot < capacity_ && keys_[slot] == key) return slot;
        return std::nullopt;
    }

    // TODO(perf): 
    // SIMD the window scan
    // broadcast key, compare 8 slots/iter (AVX-512) or 4 (AVX2), 
    // mask gaps with bits_, tzcnt the first hit
    // read path last mile
    std::optional<std::size_t> find_in(Key key, std::size_t lo_slot, std::size_t hi_slot) const {
        LI_ASSERT(lo_slot <= hi_slot && hi_slot < capacity_);

        for (std::size_t slot = lo_slot; slot <= hi_slot; ++slot) {
            if (!is_occupied(slot)) continue;
            if (keys_[slot] == key) return slot;
            if (keys_[slot] > key) return std::nullopt;
        }

        return std::nullopt;
    }

    // Never grows past the ceiling. If no window can absorb and we are already at the ceiling the
    // caller failed to capsplit
    // TODO(perf): 
    // predict pos from the segment model instead of lower_bound
    // and compute occupied_in incrementally as the window doubles (recomputed per level today)
    EditResult insert(Key key, Payload payload) {
        std::size_t pos = lower_bound(key);
        LI_ASSERT(pos == capacity_ || keys_[pos] != key);

        std::size_t height = tree_height();
        std::size_t pos_window = (pos == capacity_) ? (num_windows() - 1) : (pos / kLeafWindow);

        for (std::size_t level = 0; level <= height; ++level) {
            std::size_t base_window = (pos_window >> level) << level;
            std::size_t window_lo = base_window * kLeafWindow;
            std::size_t window_hi = window_lo + (kLeafWindow << level);

            double density = double(occupied_in(window_lo, window_hi) + 1) / double(window_hi - window_lo);

            if (density <= max_density(level, height)) {
                std::size_t placed = rebalance_with(window_lo, window_hi, key, payload);
                ++count_;
                return EditResult{ placed, window_lo, window_hi, true };
            }
        }

        if (grow()) {
            EditResult result = insert(key, payload);
            result.moved_lo = 0;
            result.moved_hi = capacity_;
            return result;
        }

        LI_ASSERT(false && "PmaBlock::insert past the ceiling, caller must cap-split at count > W_s");
        LI_ASSERT(count_ < capacity_);

        std::size_t placed = rebalance_with(0, capacity_, key, payload);
        ++count_;
        return EditResult{ placed, 0, capacity_, true };
    }

    // fast path drops into a trailing gap and relocates nothing
    // slow path defers to insert.
    // TODO(perf): 
    // cache last_occupied as a member for O(1) 
    // fast path we can remove fenwick decent and 64 ct loop
    EditResult append(Key key, Payload payload) {
        LI_ASSERT(count_ == 0 || key > keys_[last_occupied()]);

        if (count_ == 0) {
            place(0, key, payload);
            fen_add(0, +1);
            ++count_;
            return EditResult{ 0, 0, 0, true };
        }

        std::size_t pos = last_occupied() + 1;

        if (pos < capacity_) {
            std::size_t window_lo = (pos / kLeafWindow) * kLeafWindow;
            double density = double(occupied_in(window_lo, window_lo + kLeafWindow) + 1) / double(kLeafWindow);

            if (density <= max_density(0, tree_height())) {
                place(pos, key, payload);
                fen_add(pos >> 6, +1);
                ++count_;
                return EditResult{ pos, 0, 0, true };
            }
        }

        return insert(key, payload);
    }

    // TODO(perf): 
    // find() is O(log C) tons of misses
    // we can model predict + find_in as in insert... just for the future gotta keep this in mind
    EditResult erase(Key key) {
        std::optional<std::size_t> slot = find(key);
        if (!slot) return EditResult{ capacity_, 0, 0, false };

        clear_bit(*slot);
        fen_add((*slot) >> 6, -1);
        --count_;

        std::size_t moved_lo = 0;
        std::size_t moved_hi = 0;

        while (capacity_ > kLeafWindow &&
               double(count_) / double(capacity_) < min_density(tree_height(), tree_height())) {
            shrink();
            moved_lo = 0;
            moved_hi = capacity_;
        }

        return EditResult{ *slot, moved_lo, moved_hi, true };
    }


    // TODO(perf): 
    // iterate set bits per word with countr_zero instead of slot
    // unfortunate the visit callback limits simd
    template <class F>
    void scan(Key lo, Key hi, F&& visit) const {
        for (std::size_t slot = lower_bound(lo); slot < capacity_; ++slot) {
            if (!is_occupied(slot)) continue;
            if (keys_[slot] >= hi) break;
            visit(keys_[slot], payloads_[slot]);
        }
    }

    void check_invariants() const {
        LI_ASSERT(fen_prefix(num_words()) == count_);
        LI_ASSERT(capacity_ <= max_capacity_);

        std::size_t counted = 0;
        for (std::size_t i = 0; i < num_words(); ++i) counted += std::popcount(bits_[i]);
        LI_ASSERT(counted == count_);

        bool first = true;
        Key prev = 0;

        for (std::size_t slot = 0; slot < capacity_; ++slot) {
            if (!is_occupied(slot)) continue;
            if (!first) LI_ASSERT(keys_[slot] > prev);
            prev = keys_[slot];
            first = false;
        }
    }

    // TODO(perf):
    // dump sorted dumps a dense array for logical stuff
    // BAD push_back PER ELEMENT
    // we can do SIMD compress using the bitmap mask
    std::pair<std::vector<Key>, std::vector<Payload>> dump_sorted() const {
        std::vector<Key> sorted_keys;
        std::vector<Payload> sorted_payloads;

        sorted_keys.reserve(count_);
        sorted_payloads.reserve(count_);

        for (std::size_t slot = next_occupied(0); slot < capacity_; slot = next_occupied(slot + 1)) {
            sorted_keys.push_back(keys_[slot]);
            sorted_payloads.push_back(payloads_[slot]);
        }

        return { sorted_keys, sorted_payloads };
    }

private:
    std::vector<Key> keys_;
    std::vector<Payload> payloads_;
    std::vector<std::uint64_t> bits_;
    std::vector<Rank> fen_;
    std::size_t count_ = 0;
    std::size_t capacity_ = 0;
    std::size_t max_capacity_ = 0;

    static constexpr std::size_t kLeafWindow = 16;

    static constexpr double kMaxDensityLeaf = 1.00;
    static constexpr double kMaxDensityRoot = 0.50;
    static constexpr double kMinDensityLeaf = 0.10;
    static constexpr double kMinDensityRoot = 0.20;
    static constexpr double kTargetDensity = 0.65;

    static std::size_t lowbit(std::size_t x) { return x & (~x + 1); }

    void fen_add(std::size_t word, long long delta) {
        for (std::size_t index = word + 1; index <= num_words(); index += lowbit(index)) {
            fen_[index] = Rank((long long)fen_[index] + delta);
        }
    }

    Rank fen_prefix(std::size_t words) const {
        Rank sum = 0;
        for (std::size_t index = words; index > 0; index -= lowbit(index)) sum += fen_[index];
        return sum;
    }

    Rank fen_word(std::size_t index) const { return fen_prefix(index + 1) - fen_prefix(index); }

    std::size_t num_words() const { return (capacity_ + 63) >> 6; }
    std::size_t num_windows() const { return capacity_ / kLeafWindow; }
    std::size_t tree_height() const { return std::countr_zero(num_windows()); }

    bool is_occupied(std::size_t slot) const { return (bits_[slot >> 6] >> (slot & 63)) & 1ull; }
    void set_bit(std::size_t slot) { bits_[slot >> 6] |= (1ull << (slot & 63)); }
    void clear_bit(std::size_t slot) { bits_[slot >> 6] &= ~(1ull << (slot & 63)); }

    // TODO(perf): 
    // right now this is per slot clear
    // we can mask the two partial end words, memset whole words between make it supa fazt
    void clear_bits(std::size_t lo, std::size_t hi) {
        for (std::size_t slot = lo; slot < hi; ++slot) clear_bit(slot);
    }

    void place(std::size_t slot, Key key, Payload payload) {
        keys_[slot] = key;
        payloads_[slot] = payload;
        set_bit(slot);
    }

    std::size_t first_occupied() const { return slot_of_rank(0); }

    std::size_t last_occupied() const {
        if (count_ == 0) return capacity_;
        return slot_of_rank(count_ - 1);
    }

    // TODO(perf): 
    // for wide windows, we can probably unroll this cause popcount is a lot
    std::size_t occupied_in(std::size_t lo, std::size_t hi) const {
        if (lo >= hi) return 0;

        std::size_t word_lo = lo >> 6;
        std::size_t word_hi = (hi - 1) >> 6;
        std::uint64_t mask_lo = ~0ull << (lo & 63);
        std::uint64_t mask_hi = ~0ull >> (63 - ((hi - 1) & 63));

        if (word_lo == word_hi) return std::popcount(bits_[word_lo] & mask_lo & mask_hi);

        std::size_t counted = std::popcount(bits_[word_lo] & mask_lo);

        for (std::size_t word = word_lo + 1; word < word_hi; ++word) counted += std::popcount(bits_[word]);

        counted += std::popcount(bits_[word_hi] & mask_hi);

        return counted;
    }

    static double max_density(std::size_t level, std::size_t height) {
        if (height == 0) return kMaxDensityLeaf;
        return kMaxDensityRoot + (kMaxDensityLeaf - kMaxDensityRoot) * double(height - level) / double(height);
    }

    static double min_density(std::size_t level, std::size_t height) {
        if (height == 0) return kMinDensityLeaf;
        return kMinDensityRoot - (kMinDensityRoot - kMinDensityLeaf) * double(height - level) / double(height);
    }

    // TODO(perf): 
    // clear_bits per slot -> can do word level memset; 
    // the strided place() writes are a scatter
    void spread(std::size_t lo, std::size_t hi,
                std::span<const Key> keys, std::span<const Payload> payloads) {
        std::size_t count = keys.size();
        std::size_t slots = hi - lo;
        LI_ASSERT(count <= slots);

        std::size_t word_lo = lo >> 6;
        std::size_t word_hi = (hi - 1) >> 6;

        clear_bits(lo, hi);

        for (std::size_t i = 0; i < count; ++i) {
            std::size_t slot = lo + (i * slots) / count;
            place(slot, keys[i], payloads[i]);
        }

        for (std::size_t word = word_lo; word <= word_hi; ++word) {
            long long updated = (long long)std::popcount(bits_[word]);
            long long previous = (long long)fen_word(word);
            if (updated != previous) fen_add(word, updated - previous);
        }
    }

    // Gathers the window's live pairs with new_key inserted in order, spreads them evenly, and
    // returns the slot where new_key landed (spread places element i at lo + (i * slots) / count)
    // TODO(perf): 
    // the gather loop is a bitmap compaction -> simd compress reading keys_/payloads_ by the bits_ mask
    std::size_t rebalance_with(std::size_t lo, std::size_t hi, Key new_key, Payload new_payload) {
        std::vector<Key> gathered_keys;
        std::vector<Payload> gathered_payloads;

        gathered_keys.reserve(occupied_in(lo, hi) + 1);
        gathered_payloads.reserve(gathered_keys.capacity());

        std::size_t insert_index = 0;
        bool inserted = false;

        for (std::size_t slot = lo; slot < hi; ++slot) {
            if (!is_occupied(slot)) continue;

            if (!inserted && keys_[slot] > new_key) {
                insert_index = gathered_keys.size();
                gathered_keys.push_back(new_key);
                gathered_payloads.push_back(new_payload);
                inserted = true;
            }

            gathered_keys.push_back(keys_[slot]);
            gathered_payloads.push_back(payloads_[slot]);
        }

        if (!inserted) {
            insert_index = gathered_keys.size();
            gathered_keys.push_back(new_key);
            gathered_payloads.push_back(new_payload);
        }

        spread(lo, hi, gathered_keys, gathered_payloads);

        std::size_t count = gathered_keys.size();
        std::size_t slots = hi - lo;

        return lo + (insert_index * slots) / count;
    }

    // TODO(perf): same bitmap guided gather as rebalance_with -> SIMD compress
    // need to copy tho 
    void resize_to(std::size_t new_capacity) {
        std::vector<Key> gathered_keys;
        std::vector<Payload> gathered_payloads;

        gathered_keys.reserve(count_);
        gathered_payloads.reserve(count_);

        for (std::size_t slot = 0; slot < capacity_; ++slot) {
            if (!is_occupied(slot)) continue;
            gathered_keys.push_back(keys_[slot]);
            gathered_payloads.push_back(payloads_[slot]);
        }

        keys_.assign(new_capacity, Key{});
        payloads_.assign(new_capacity, Payload{});
        capacity_ = new_capacity;
        bits_.assign(num_words(), 0);
        fen_.assign(num_words() + 1, 0);

        spread(0, new_capacity, gathered_keys, gathered_payloads);
    }

    bool grow() {
        std::size_t target = capacity_ * 2;
        if (target > max_capacity_) target = max_capacity_;
        if (target == capacity_) return false;

        resize_to(target);
        return true;
    }

    void shrink() { resize_to(capacity_ / 2); }

    // TODO(perf): 64-trip loop -> PDEP(1ull << k, word) then countr_zero for O(!)
    static std::size_t select_in_word(std::uint64_t word, std::size_t k) {
        for (std::size_t bit = 0; bit < 64; ++bit) {
            if (word & 1ull) {
                if (k == 0) return bit;
                --k;
            }
            word >>= 1;
        }
        return 64;
    }

    static std::size_t choose_capacity(std::size_t count, std::size_t max_capacity_slots) {
        std::size_t needed = (count == 0) ? 1 : std::size_t(std::ceil(double(count) / kTargetDensity));
        if (needed < kLeafWindow) needed = kLeafWindow;

        std::size_t windows = (needed + kLeafWindow - 1) / kLeafWindow;
        std::size_t capacity = kLeafWindow * std::bit_ceil(windows);

        return (capacity < max_capacity_slots) ? capacity : max_capacity_slots;
    }
};

}