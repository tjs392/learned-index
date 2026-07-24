/*
* segment_store.hpp - Per segment store, dense sorted array of keys and payloads
*
* Each segment owns one block, keys and payloads in a single alloc
* [0, count) is sorted and gap free, so rank == slot index
*
*
* Worst Case Bounds:
* insert/erase: binary search plus one memmove of at most count entries, count <= w_s
* every op completes inline, no deferral, no fallback
*
*/

#pragma once

#include "status.hpp"

#include <vector>
#include <span>
#include <cstring>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <algorithm>
#include <memory>
#include <type_traits>

#if defined(CEDAR_PHASE_TIMING)
#include <chrono>
    namespace li::detail::store_phase {
    enum Q { 
        kLowerBound, 
        kSpread, 
        kQN 
    };

    inline const char* qname(int q) {
        static const char* n[] = { "  store.lower_bound", "  store.spread" };
        return n[q];
    }

    inline std::vector<double> qs[kQN];

    struct QScope {
        int q;
        std::chrono::steady_clock::time_point t0;

        explicit QScope(int q_) : q(q_), t0(std::chrono::steady_clock::now()) {}

        ~QScope() {
            qs[q].push_back(
                std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count());
        }
    };

    inline void qreset() {
        for (int i = 0; i < kQN; ++i) qs[i].clear();
    }
    }

    #define CEDAR_PP_CAT2(a, b) a##b
    #define CEDAR_PP_CAT(a, b) CEDAR_PP_CAT2(a, b)
    #define CEDAR_STORE_PHASE(q) ::li::detail::store_phase::QScope CEDAR_PP_CAT(_li_qs_, __LINE__)(::li::detail::store_phase::q)
#else
    #define CEDAR_STORE_PHASE(q) ((void)0)
#endif

namespace li::detail {

// vector allocator that skips zero initialization ofm its slots
template <class T, class A = std::allocator<T>>
struct default_init_alloc : A {

    template <class U> 
    struct rebind {
        using other = default_init_alloc<U, typename std::allocator_traits<A>::template rebind_alloc<U>>;
    };

    using A::A;

    template <class U> 
    void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) {
        ::new (static_cast<void*>(p)) U;
    }

    template <class U, class... Args> 
    void construct(U* p, Args&&... args) {
        std::allocator_traits<A>::construct(static_cast<A&>(*this), p, std::forward<Args>(args)...);
    }
};

template <class T> using raw_vector = std::vector<T, default_init_alloc<T>>;

class SegmentStore {
public:

    struct EditResult {
        std::size_t slot;
        bool found;
    };

    static std::size_t capacity_for_key_cap(std::size_t key_cap) {
        std::size_t needed = std::size_t(std::ceil(double(key_cap + 1) / kTargetFill));

        return needed < kMinCapacity ? kMinCapacity : needed;
    }

    static SegmentStore bulk_load(std::span<const Key> keys, 
                                  std::span<const Payload> payloads,
                                  std::size_t max_capacity_slots) {
        CEDAR_ASSERT(keys.size() == payloads.size());

        SegmentStore block;
        block.max_capacity_ = max_capacity_slots;
        block.capacity_ = choose_capacity(keys.size(), max_capacity_slots);


        CEDAR_CHECK(keys.size() <= block.capacity_,
                 "SegmentStore::bulk_load: %zu keys > %zu slots (ceiling clamp; w_s >= w_m violated "
                 "or an install site overflowed the cap)",
                 keys.size(), block.capacity_);

        block.alloc_fresh(block.capacity_);
        block.count_ = keys.size();

        if (!keys.empty()) {
            std::memcpy(block.key_data(), keys.data(), keys.size() * sizeof(Key));
            std::memcpy(block.payload_data(), payloads.data(), payloads.size() * sizeof(Payload));
        }

        return block;
    }

    std::size_t size() const { return count_; }
    std::size_t capacity() const { return capacity_; }
    std::size_t max_capacity() const { return max_capacity_; }
    bool empty() const { return count_ == 0; }

    Key max_key() const {
        CEDAR_ASSERT(count_ > 0);

        return key_data()[count_ - 1];
    }


    Rank local_rank(std::size_t slot) const { return slot < count_ ? slot : count_; }
    std::size_t slot_of_rank(Rank rank) const { return rank < count_ ? std::size_t(rank) : capacity_; }

    Key key_at(std::size_t slot) const {
        CEDAR_ASSERT(slot < count_);
        return key_data()[slot];
    }

    Payload payload_at(std::size_t slot) const {
        CEDAR_ASSERT(slot < count_);
        return payload_data()[slot];
    }

    
    std::size_t lower_bound(Key key) const {
        std::size_t lo = 0, hi = count_;

        while (lo < hi) {
            std::size_t mid = lo + (hi - lo) / 2;

            if (key_data()[mid] < key) lo = mid + 1;
            else hi = mid;
        }

        return lo < count_ ? lo : capacity_;
    }

    std::optional<std::size_t> find(Key key) const {
        std::size_t slot = lower_bound(key);

        if (slot < capacity_ && key_data()[slot] == key) return slot;

        return std::nullopt;
    }

    std::optional<std::size_t> find_in(Key key, std::size_t lo_slot, std::size_t hi_slot) const {
        CEDAR_ASSERT(lo_slot <= hi_slot && hi_slot < capacity_);

        const std::size_t hi = hi_slot < count_ ? hi_slot : (count_ ? count_ - 1 : 0);

        for (std::size_t slot = lo_slot; slot <= hi && slot < count_; ++slot) {
            if (key_data()[slot] == key) return slot;
            if (key_data()[slot] > key) return std::nullopt;
        }

        return std::nullopt;
    }

    EditResult insert(Key key, Payload payload) {
        std::size_t pos;

        {
            CEDAR_STORE_PHASE(kLowerBound);
            pos = lower_bound(key);
        }

        if (pos == capacity_) pos = count_;

        CEDAR_ASSERT(pos == count_ || key_data()[pos] != key);

        if (count_ == capacity_ && !grow()) {
            CEDAR_CHECK(false,
                     "SegmentStore::insert past the ceiling (count=%zu == max_capacity=%zu): "
                     "caller must cap-split at count > w_s",
                     count_, max_capacity_);
        }

        {
            CEDAR_STORE_PHASE(kSpread);

            if (pos < count_) {
                std::memmove(key_data() + pos + 1, key_data() + pos, (count_ - pos) * sizeof(Key));
                std::memmove(payload_data() + pos + 1, payload_data() + pos, (count_ - pos) * sizeof(Payload));
            }

            key_data()[pos] = key;
            payload_data()[pos] = payload;
        }

        ++count_;

        return EditResult{ pos, true };
    }

    EditResult append(Key key, Payload payload) {
        CEDAR_ASSERT(count_ == 0 || key > key_data()[count_ - 1]);

        if (count_ == capacity_ && !grow()) {
            CEDAR_CHECK(false,
                     "SegmentStore::append past the ceiling (count=%zu == max_capacity=%zu): "
                     "caller must cap-split at count > w_s",
                     count_, max_capacity_);
        }

        key_data()[count_] = key;
        payload_data()[count_] = payload;
        ++count_;

        return EditResult{ count_ - 1, true };
    }

    EditResult erase(Key key) {
        std::optional<std::size_t> slot = find(key);

        if (!slot) return EditResult{ capacity_, false };

        const std::size_t pos = *slot;

        if (pos + 1 < count_) {
            std::memmove(key_data() + pos, key_data() + pos + 1, (count_ - pos - 1) * sizeof(Key));
            std::memmove(payload_data() + pos, payload_data() + pos + 1, (count_ - pos - 1) * sizeof(Payload));
        }

        --count_;

        // bounded space under deletes
        if (capacity_ > kMinCapacity && count_ < capacity_ / 4) {
            resize_to(std::max(kMinCapacity, capacity_ / 2));
        }

        return EditResult{ pos, true };
    }

    template <class F>
    void scan(Key lo, Key hi, F&& visit) const {
        std::size_t slot = lower_bound(lo);

        if (slot == capacity_) return;

        for (; slot < count_ && key_data()[slot] < hi; ++slot) {
            visit(key_data()[slot], payload_data()[slot]);
        }
    }

    void check_invariants() const {
        CEDAR_ASSERT(count_ <= capacity_);
        CEDAR_ASSERT(capacity_ <= std::max(max_capacity_, kMinCapacity));

        for (std::size_t slot = 1; slot < count_; ++slot) CEDAR_ASSERT(key_data()[slot] > key_data()[slot - 1]);
    }


    // **IMPORTANT**
    // these spans alias this block's storage, finish building every replacement
    // segment before this block is destroyed
    std::span<const Key> keys_view() const { return { key_data(), count_ }; }
    std::span<const Payload> payloads_view() const { return { payload_data(), count_ }; }

    static SegmentStore bulk_load2(std::span<const Key> k1, 
                                   std::span<const Payload> p1,
                                   std::span<const Key> k2, 
                                   std::span<const Payload> p2,
                                   std::size_t max_capacity_slots) {
        CEDAR_ASSERT(k1.size() == p1.size() && k2.size() == p2.size());
        CEDAR_ASSERT(k1.empty() || k2.empty() || k1.back() < k2.front());

        const std::size_t total = k1.size() + k2.size();

        SegmentStore block;
        block.max_capacity_ = max_capacity_slots;
        block.capacity_ = choose_capacity(total, max_capacity_slots);

        CEDAR_CHECK(total <= block.capacity_,
                 "SegmentStore::bulk_load2: %zu keys > %zu slots (ceiling clamp; w_s >= w_m violated "
                 "or an install site overflowed the cap)",
                 total, block.capacity_);

        block.alloc_fresh(block.capacity_);
        block.count_ = total;

        if (!k1.empty()) {
            std::memcpy(block.key_data(), k1.data(), k1.size() * sizeof(Key));
            std::memcpy(block.payload_data(), p1.data(), p1.size() * sizeof(Payload));
        }

        if (!k2.empty()) {
            std::memcpy(block.key_data() + k1.size(), k2.data(), k2.size() * sizeof(Key));
            std::memcpy(block.payload_data() + k1.size(), p2.data(), p2.size() * sizeof(Payload));
        }

        return block;
    }


private:

    static_assert(std::is_same_v<Key, Payload>, "combined layout stores payloads in the key buffer");

    raw_vector<Key> buf_;

    Key* key_data() { return buf_.data(); }
    const Key* key_data() const { return buf_.data(); }

    Payload* payload_data() { return buf_.data() + capacity_; }
    const Payload* payload_data() const { return buf_.data() + capacity_; }

    std::size_t count_ = 0;
    std::size_t capacity_ = 0;
    std::size_t max_capacity_ = 0;

    static constexpr std::size_t kMinCapacity = 16;


    #ifndef CEDAR_DENSE_TARGET_FILL
        #define CEDAR_DENSE_TARGET_FILL 0.90
    #endif

    static constexpr double kTargetFill = CEDAR_DENSE_TARGET_FILL;

    static std::size_t choose_capacity(std::size_t count, std::size_t max_capacity_slots) {
        std::size_t needed = (count == 0) ? kMinCapacity : std::size_t(std::ceil(double(count) / kTargetFill));

        if (needed < kMinCapacity) needed = kMinCapacity;

        return needed < max_capacity_slots ? needed : max_capacity_slots;
    }

    // fresh block at exact capacity, resize from the empty state is exact
    void alloc_fresh(std::size_t cap) { buf_.resize(2 * cap); }

    void resize_to(std::size_t new_capacity) {
        CEDAR_ASSERT(new_capacity >= count_);

        if (new_capacity > capacity_) {
            // note; vector::resize rounds capacity up and quietly turns any grow factor under 2x back into 2x
            raw_vector<Key> fresh;

            fresh.reserve(2 * new_capacity);
            fresh.resize(2 * new_capacity);

            std::memcpy(fresh.data(), buf_.data(), count_ * sizeof(Key));
            std::memcpy(fresh.data() + new_capacity, buf_.data() + capacity_, count_ * sizeof(Payload));

            buf_ = std::move(fresh);
        } else {
            std::memmove(buf_.data() + new_capacity, buf_.data() + capacity_, count_ * sizeof(Payload));
            buf_.resize(2 * new_capacity);
        }

        capacity_ = new_capacity;
    }

    // 1.25x. the grow factor sets steady state fill under churn, 2x leaves too much slack
    // TODO: parametize this
    bool grow() {
        std::size_t target = capacity_ + capacity_ / 4;

        if (target > max_capacity_) target = max_capacity_;
        if (target <= capacity_) return false;

        resize_to(target);

        return true;
    }
};

}