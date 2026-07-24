// test_segment_store.cpp - differential harness for the dense store: every
// observable vs a sorted-vector reference, invariants after every op, and the
// always on ceilings

#include <gtest/gtest.h>

#include "cedar/segment_store.hpp"

#include <vector>
#include <utility>
#include <random>
#include <optional>
#include <algorithm>

using namespace li;
using li::detail::SegmentStore;

namespace {

constexpr std::size_t kKeyCap = 512;

struct Ref {
    std::vector<std::pair<Key, Payload>> v;

    std::size_t lb(Key k) const {
        std::size_t lo = 0;
        while (lo < v.size() && v[lo].first < k) ++lo;
        return lo;
    }

    bool has(Key k) const {
        const std::size_t p = lb(k);
        return p < v.size() && v[p].first == k;
    }
};

SegmentStore make(const Ref& ref, std::size_t max_slots, std::mt19937_64& rng) {
    std::vector<Key> ks(ref.v.size());
    std::vector<Payload> ps(ref.v.size());

    for (std::size_t i = 0; i < ref.v.size(); ++i) {
        ks[i] = ref.v[i].first;
        ps[i] = ref.v[i].second;
    }

    if (ks.size() >= 2 && (rng() % 3) == 0) {
        const std::size_t cut = 1 + rng() % (ks.size() - 1);
        return SegmentStore::bulk_load2({ ks.data(), cut }, { ps.data(), cut },
                                        { ks.data() + cut, ks.size() - cut },
                                        { ps.data() + cut, ps.size() - cut }, max_slots);
    }

    return SegmentStore::bulk_load({ ks.data(), ks.size() }, { ps.data(), ps.size() }, max_slots);
}

void check_state(const SegmentStore& st, const Ref& ref, std::mt19937_64& rng) {
    st.check_invariants();

    ASSERT_EQ(st.size(), ref.v.size());

    if (!ref.v.empty()) {
        EXPECT_EQ(st.max_key(), ref.v.back().first);
    }

    for (std::size_t t = 0; t < 4 && !ref.v.empty(); ++t) {
        const std::size_t s = rng() % ref.v.size();

        EXPECT_EQ(st.key_at(s), ref.v[s].first);
        EXPECT_EQ(st.payload_at(s), ref.v[s].second);
        EXPECT_EQ(st.local_rank(s), Rank(s));
        EXPECT_EQ(st.slot_of_rank(Rank(s)), s);
    }

    const Key probe = 1 + rng() % 60000;
    const std::size_t p = ref.lb(probe);
    const std::size_t got = st.lower_bound(probe);

    if (p < ref.v.size()) {
        EXPECT_EQ(got, p);
    } else {
        EXPECT_EQ(got, st.capacity());
    }

    std::optional<std::size_t> f = st.find(probe);

    if (ref.has(probe)) {
        ASSERT_TRUE(f.has_value());
        EXPECT_EQ(*f, p);
    } else {
        EXPECT_FALSE(f.has_value());
    }
}

}

TEST(SegmentStore, BulkLoad2MatchesConcatenatedBulkLoad) {
    std::mt19937_64 rng(11);
    const std::size_t max_slots = SegmentStore::capacity_for_key_cap(kKeyCap);

    std::vector<Key> ks;
    std::vector<Payload> ps;
    Key k = 5;

    for (int i = 0; i < 300; ++i) {
        ks.push_back(k);
        ps.push_back(Payload(rng()));
        k += 1 + rng() % 30;
    }

    SegmentStore whole =
        SegmentStore::bulk_load({ ks.data(), ks.size() }, { ps.data(), ps.size() }, max_slots);

    for (std::size_t cut : { std::size_t(1), std::size_t(150), std::size_t(299) }) {
        SegmentStore two =
            SegmentStore::bulk_load2({ ks.data(), cut }, { ps.data(), cut },
                                     { ks.data() + cut, ks.size() - cut },
                                     { ps.data() + cut, ps.size() - cut }, max_slots);

        two.check_invariants();

        ASSERT_EQ(two.size(), whole.size());
        EXPECT_EQ(two.capacity(), whole.capacity());

        std::span<const Key> wk = whole.keys_view();
        std::span<const Key> tk = two.keys_view();
        std::span<const Payload> wp = whole.payloads_view();
        std::span<const Payload> tp = two.payloads_view();

        for (std::size_t i = 0; i < whole.size(); ++i) {
            EXPECT_EQ(tk[i], wk[i]);
            EXPECT_EQ(tp[i], wp[i]);
        }
    }
}

TEST(SegmentStore, SentinelContracts) {
    std::mt19937_64 rng(12);
    const std::size_t max_slots = SegmentStore::capacity_for_key_cap(kKeyCap);

    Ref ref;
    ref.v = { { 100, 1 }, { 200, 2 }, { 300, 3 } };
    SegmentStore st = make(ref, max_slots, rng);

    EXPECT_EQ(st.lower_bound(1000), st.capacity());
    EXPECT_EQ(st.slot_of_rank(Rank(3)), st.capacity());
    EXPECT_FALSE(st.find(150).has_value());
    EXPECT_FALSE(st.find_in(150, 0, st.capacity() - 1).has_value());

    SegmentStore::EditResult miss = st.erase(150);

    EXPECT_FALSE(miss.found);
    EXPECT_EQ(miss.slot, st.capacity());

    SegmentStore empty = SegmentStore::bulk_load({}, {}, max_slots);

    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(empty.lower_bound(1), empty.capacity());
    EXPECT_FALSE(empty.find(1).has_value());
}

TEST(SegmentStore, ScanIsHalfOpen) {
    std::mt19937_64 rng(13);
    const std::size_t max_slots = SegmentStore::capacity_for_key_cap(kKeyCap);

    Ref ref;
    ref.v = { { 10, 1 }, { 20, 2 }, { 30, 3 }, { 40, 4 } };
    SegmentStore st = make(ref, max_slots, rng);

    std::vector<Key> got;
    st.scan(20, 40, [&](Key k, Payload) { got.push_back(k); });

    EXPECT_EQ(got, (std::vector<Key>{ 20, 30 }));

    got.clear();
    st.scan(41, 100, [&](Key k, Payload) { got.push_back(k); });

    EXPECT_TRUE(got.empty());
}

TEST(SegmentStore, DifferentialChurn) {
    std::mt19937_64 rng(20260724);
    const std::size_t max_slots = SegmentStore::capacity_for_key_cap(kKeyCap);

    for (int round = 0; round < 60; ++round) {
        Ref ref;
        Key k = 1 + rng() % 50;
        const std::size_t seed_n = (round % 7 == 0) ? std::size_t(round % 3)
                                                    : std::size_t(rng() % 300);

        for (std::size_t i = 0; i < seed_n; ++i) {
            ref.v.push_back({ k, Payload(rng()) });
            k += 1 + rng() % 40;
        }

        SegmentStore st = make(ref, max_slots, rng);

        for (int op = 0; op < 700; ++op) {
            const unsigned dice = unsigned(rng() % 100);

            if (dice < 40 && ref.v.size() < kKeyCap) {
                Key nk = 1 + rng() % 60000;
                if (ref.has(nk)) continue;

                SegmentStore::EditResult e = st.insert(nk, Payload(nk * 3));

                EXPECT_TRUE(e.found);
                EXPECT_EQ(e.slot, ref.lb(nk));
                ref.v.insert(ref.v.begin() + std::ptrdiff_t(ref.lb(nk)), { nk, Payload(nk * 3) });
            } else if (dice < 55 && ref.v.size() < kKeyCap) {
                const Key nk = (ref.v.empty() ? 0 : ref.v.back().first) + 1 + rng() % 30;
                SegmentStore::EditResult e = st.append(nk, Payload(nk * 5));

                EXPECT_TRUE(e.found);
                EXPECT_EQ(e.slot, ref.v.size());
                ref.v.push_back({ nk, Payload(nk * 5) });
            } else if (!ref.v.empty()) {
                Key victim;
                if (dice < 85) victim = ref.v[rng() % ref.v.size()].first;
                else victim = 1 + rng() % 60000;

                const bool had = ref.has(victim);
                const std::size_t want = had ? ref.lb(victim) : 0;
                SegmentStore::EditResult e = st.erase(victim);

                EXPECT_EQ(e.found, had);

                if (had) {
                    EXPECT_EQ(e.slot, want);
                    ref.v.erase(ref.v.begin() + std::ptrdiff_t(want));
                }
            }

            check_state(st, ref, rng);

            if (::testing::Test::HasFailure()) return;
        }

        std::span<const Key> kv = st.keys_view();
        std::span<const Payload> pv = st.payloads_view();

        ASSERT_EQ(kv.size(), ref.v.size());

        for (std::size_t i = 0; i < kv.size(); ++i) {
            EXPECT_EQ(kv[i], ref.v[i].first);
            EXPECT_EQ(pv[i], ref.v[i].second);
        }
    }
}

TEST(SegmentStore, ShrinkKeepsSpaceBounded) {
    std::mt19937_64 rng(14);
    const std::size_t max_slots = SegmentStore::capacity_for_key_cap(kKeyCap);

    Ref ref;
    Key k = 10;

    for (int i = 0; i < int(kKeyCap); ++i) {
        ref.v.push_back({ k, Payload(k) });
        k += 3;
    }

    SegmentStore st = make(ref, max_slots, rng);
    const std::size_t full_cap = st.capacity();

    while (ref.v.size() > 4) {
        const Key victim = ref.v[rng() % ref.v.size()].first;
        const std::size_t p = ref.lb(victim);

        st.erase(victim);
        ref.v.erase(ref.v.begin() + std::ptrdiff_t(p));
        st.check_invariants();
    }

    EXPECT_LT(st.capacity(), full_cap / 4);
    check_state(st, ref, rng);
}

TEST(SegmentStoreDeath, AppendPastCeilingTripsAlwaysOnCheck) {
    SegmentStore st = SegmentStore::bulk_load({}, {}, 16);

    for (int i = 0; i < 16; ++i) st.append(Key(10 + i), Payload(i));

    EXPECT_DEATH(st.append(1000, 999), "append past the ceiling");
}

TEST(SegmentStoreDeath, InsertPastCeilingTripsAlwaysOnCheck) {
    SegmentStore st = SegmentStore::bulk_load({}, {}, 16);

    for (int i = 0; i < 16; ++i) st.append(Key(10 + 2 * i), Payload(i));

    EXPECT_DEATH(st.insert(11, 1), "insert past the ceiling");
}

TEST(SegmentStoreDeath, BulkLoadOverCeilingTripsAlwaysOnCheck) {
    std::vector<Key> ks;
    std::vector<Payload> ps;

    for (int i = 0; i < 40; ++i) {
        ks.push_back(Key(10 + i));
        ps.push_back(Payload(i));
    }

    EXPECT_DEATH(
        SegmentStore::bulk_load({ ks.data(), ks.size() }, { ps.data(), ps.size() }, 16),
        "keys > ");
}
