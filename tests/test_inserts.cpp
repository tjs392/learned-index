#include <gtest/gtest.h>
#include "li/index.hpp"
#include "li/status.hpp"
#include "li/model.hpp"

#include <vector>
#include <set>
#include <map>
#include <random>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace {

using li::LearnedIndex;
using li::Status;
using li::Key;
using li::Payload;
using li::Rank;

constexpr double kEpsSlack = 1e-9;

void check_index(const LearnedIndex& idx,
                 const std::set<Key>& live,
                 const std::map<Key, Payload>& payloads,
                 double eps) {
    const auto& mt = idx.mapping_table_for_test();
    const auto& bl = idx.blocks_for_test();
    ASSERT_EQ(mt.size(), bl.size()) << "descriptor/block vectors misaligned";

    if (live.empty()) { EXPECT_TRUE(mt.empty()); return; }
    ASSERT_FALSE(mt.empty());

    std::vector<Rank> base(mt.size(), 0);
    Rank acc = 0;
    for (size_t i = 0; i < mt.size(); ++i) {
        base[i] = acc;
        EXPECT_GT(mt[i].count, 0u) << "empty segment at " << i;
        EXPECT_EQ(mt[i].count, bl[i].size()) << "count != block size at " << i;
        if (i) EXPECT_LT(mt[i - 1].key_low, mt[i].key_low)
            << "key_low not strictly increasing at " << i;
        bl[i].check_invariants();
        acc += mt[i].count;
    }
    EXPECT_EQ(acc, live.size()) << "sum of counts != live key count";

    Rank gi = 0;
    for (Key k : live) {
        const size_t i = idx.find_descriptor(k);
        const auto& d = mt[i];
        EXPECT_LE(d.key_low, k) << "key " << k << " below its segment key_low";
        if (i + 1 < mt.size())
            EXPECT_LT(k, mt[i + 1].key_low) << "key " << k << " leaked past its segment";

        const Rank local = gi - base[i];
        const double pred = d.model.alpha * double(k - d.key_low) + d.model.beta;
        EXPECT_LE(std::fabs(pred - double(local)), eps + kEpsSlack)
            << "HARD-EPS VIOLATION key " << k << " seg " << i
            << " local_rank " << local << " pred " << pred << " eps " << eps;

        auto gr = idx.global_rank_of(k);
        ASSERT_TRUE(gr.ok()) << "global_rank_of missing live key " << k;
        EXPECT_EQ(gr.value(), gi) << "wrong derived rank for key " << k;

        auto pl = idx.point_lookup(k);
        ASSERT_TRUE(pl.ok()) << "live key " << k << " not found";
        EXPECT_EQ(pl.value(), payloads.at(k)) << "wrong payload for key " << k;
        ++gi;
    }

    std::vector<Key> scanned;
    for (Key k : idx.range_lookup(*live.begin(), *live.rbegin())) scanned.push_back(k);
    EXPECT_EQ(scanned, std::vector<Key>(live.begin(), live.end()))
        << "range scan mismatch after structural change";
}

LearnedIndex build_index(const std::vector<Key>& sorted_keys, double eps,
                         std::set<Key>& live, std::map<Key, Payload>& payloads) {
    LearnedIndex idx(eps);
    idx.build(sorted_keys);
    live.clear();
    payloads.clear();
    for (uint64_t i = 0; i < sorted_keys.size(); ++i) {
        live.insert(sorted_keys[i]);
        payloads[sorted_keys[i]] = static_cast<Payload>(i);
    }
    return idx;
}

void insert_and_check(LearnedIndex& idx, Key k, Payload p,
                      std::set<Key>& live, std::map<Key, Payload>& payloads, double eps) {
    ASSERT_EQ(live.count(k), 0u) << "test bug: duplicate insert of " << k;
    const size_t before = idx.mapping_table_for_test().size();

    ASSERT_EQ(idx.insert(k, p), Status::ok);
    live.insert(k);
    payloads[k] = p;

    const size_t after = idx.mapping_table_for_test().size();
    const long long delta =
        static_cast<long long>(after) - static_cast<long long>(before);
    EXPECT_LE(std::llabs(delta), 2)
        << "one op moved segment count by " << delta
        << " (L3 caps abs(delta) at 2: split +<=2, restore merge -<=2)";
        
    check_index(idx, live, payloads, eps);
}

TEST(LearnedIndexInsert, BatteryHoldsOnFreshBuild) {
    std::vector<Key> keys;
    uint64_t v = 0;
    for (uint64_t i = 0; i < 200; ++i) { v += 1 + (i % 9); keys.push_back(v); }
    std::set<Key> live; std::map<Key, Payload> payloads;
    LearnedIndex idx = build_index(keys, 1.0, live, payloads);
    check_index(idx, live, payloads, 1.0);
}

TEST(LearnedIndexInsert, IntoEmptySeedsOneSegment) {
    LearnedIndex idx(4.0);
    std::set<Key> live; std::map<Key, Payload> payloads;
    insert_and_check(idx, 500, 12345, live, payloads, 4.0);
    EXPECT_EQ(idx.mapping_table_for_test().size(), 1u);
    EXPECT_EQ(idx.point_lookup(500).value(), 12345u);
}

TEST(LearnedIndexInsert, CoverableKeyRefitsNoSplit) {
    std::vector<Key> keys;
    for (uint64_t i = 0; i < 200; ++i) keys.push_back(1000 + 2 * i);
    std::set<Key> live; std::map<Key, Payload> payloads;
    LearnedIndex idx = build_index(keys, 4.0, live, payloads);
    ASSERT_EQ(idx.mapping_table_for_test().size(), 1u);
    insert_and_check(idx, 1001, 77, live, payloads, 4.0);
    EXPECT_EQ(idx.mapping_table_for_test().size(), 1u) << "coverable insert must not split";
}

TEST(LearnedIndexInsert, InteriorInsertForcesTwoPieces) {
    std::vector<Key> keys = {0, 10, 20, 30, 40};
    std::set<Key> live; std::map<Key, Payload> payloads;
    LearnedIndex idx = build_index(keys, 0.0, live, payloads);
    ASSERT_EQ(idx.mapping_table_for_test().size(), 1u);
    insert_and_check(idx, 5, 7, live, payloads, 0.0);
    EXPECT_EQ(idx.mapping_table_for_test().size(), 2u);
}

TEST(LearnedIndexInsert, InsertRaisesCountByTwo) {
    std::vector<Key> keys = {0, 10, 20, 30, 40};
    std::set<Key> live; std::map<Key, Payload> payloads;
    LearnedIndex idx = build_index(keys, 0.0, live, payloads);
    ASSERT_EQ(idx.mapping_table_for_test().size(), 1u) << "exact line should build to one segment";
    check_index(idx, live, payloads, 0.0);
    insert_and_check(idx, 15, 424242, live, payloads, 0.0);
    EXPECT_EQ(idx.mapping_table_for_test().size(), 3u) << "off-trend insert must force 1 -> 3";
}

TEST(LearnedIndexInsert, FrontInsertShiftsAllRanks) {
    std::vector<Key> keys;
    for (uint64_t i = 0; i < 20; ++i) keys.push_back(100 + 10 * i);
    std::set<Key> live; std::map<Key, Payload> payloads;
    LearnedIndex idx = build_index(keys, 2.0, live, payloads);
    insert_and_check(idx, 50, 314159, live, payloads, 2.0);

    EXPECT_EQ(idx.mapping_table_for_test().front().key_low, 50u)
        << "front key_low must update to the new minimum";
    EXPECT_EQ(idx.global_rank_of(50).value(), 0u);
    EXPECT_EQ(idx.global_rank_of(100).value(), 1u) << "old min must shift to rank 1";
}

TEST(LearnedIndexInsert, BackInsertAppends) {
    std::vector<Key> keys;
    for (uint64_t i = 0; i < 20; ++i) keys.push_back(100 + 10 * i);
    std::set<Key> live; std::map<Key, Payload> payloads;
    LearnedIndex idx = build_index(keys, 2.0, live, payloads);
    const Key last = keys.back();
    insert_and_check(idx, last + 1000, 271828, live, payloads, 2.0);
    EXPECT_EQ(idx.global_rank_of(last + 1000).value(), keys.size());
}

TEST(LearnedIndexInsert, PayloadStaysOpaqueWhileRankDerives) {
    std::vector<Key> keys = {10, 20, 30, 40, 50};
    std::set<Key> live; std::map<Key, Payload> payloads;
    LearnedIndex idx = build_index(keys, 2.0, live, payloads);
    insert_and_check(idx, 25, 999, live, payloads, 2.0);

    EXPECT_EQ(idx.point_lookup(25).value(), 999u) << "inserted payload returned verbatim";
    EXPECT_EQ(idx.global_rank_of(25).value(), 2u);
    EXPECT_EQ(idx.point_lookup(30).value(), 2u) << "stored payload frozen (old build rank)";
    EXPECT_EQ(idx.global_rank_of(30).value(), 3u) << "derived rank tracks the shift";
}

TEST(LearnedIndexInsert, RandomChurnMaintainsInvariants) {
    for (double eps : {0.5, 1.0, 2.0, 8.0}) {
        std::mt19937_64 rng(12345 + uint64_t(eps * 10));
        std::vector<Key> base_keys;
        uint64_t v = 0;
        for (uint64_t i = 0; i < 80; ++i) { v += 1 + (rng() % 20); base_keys.push_back(v); }
        std::set<Key> live; std::map<Key, Payload> payloads;
        LearnedIndex idx = build_index(base_keys, eps, live, payloads);
        check_index(idx, live, payloads, eps);

        std::uniform_int_distribution<uint64_t> key_dist(1, v + 500);
        for (int n = 0; n < 250; ++n) {
            Key k = key_dist(rng);
            while (live.count(k)) k = key_dist(rng);
            insert_and_check(idx, k, 1000000000ull + uint64_t(n), live, payloads, eps);
        }
    }
}

TEST(LearnedIndexInsert, BuildsFromEmptyViaInserts) {
    const double eps = 1.0;
    std::mt19937_64 rng(999);
    LearnedIndex idx(eps);
    std::set<Key> live; std::map<Key, Payload> payloads;
    std::uniform_int_distribution<uint64_t> key_dist(1, 100000);
    for (int n = 0; n < 300; ++n) {
        Key k = key_dist(rng);
        while (live.count(k)) k = key_dist(rng);
        insert_and_check(idx, k, k + 500000, live, payloads, eps);
    }
    EXPECT_GE(idx.mapping_table_for_test().size(), 1u);
}

#ifdef LI_INVARIANT_CHECKS
TEST(LearnedIndexInsertDeath, DuplicateKeyTripsAssert) {
    LearnedIndex idx(2.0);
    idx.build(std::vector<Key>{10, 20, 30});
    EXPECT_DEATH({ idx.insert(20, 0); }, "");
}
#endif

}