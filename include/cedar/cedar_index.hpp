/*
* cedar_index.hpp - The learned index
*
* One mapping table of Segments, each a model plus an owned SegmentStore and a DriftGuard
* every insert and delete runs the cheap band feasibility check first and only falls back to the exact
* O(W_s) cone (minimal_line_cover) on a break
* Segments are capped at W_s keys and we do cap split before overflow
* merged when a neighbor pair fits under W_m, and each carries a refit budget B that
* force splits instead of thrashing
* Important: no cascade since theres a bounded number of structural ops per operation.
*/


#pragma once

#include "status.hpp"
#include "segment_store.hpp"
#include "segmentation.hpp"
#include "drift_guard.hpp"
#include "segment_tree.hpp"
#include <vector>
#include <chrono>
#include <memory>
#include <span>
#include <algorithm>
#include <cmath>
#include <utility>
#include <limits>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdint>

namespace li {

// for bench marking the timing of phases
#if defined(CEDAR_PHASE_TIMING)
    namespace phase {a
    enum P { kRebuildTotal, kApplyRefit, kDumpSorted, kLineCover, kPieceSplit, kMiddleSplit,
            kMergeFlanksInRebuild, kMergeFlanksInInsert, kStoreInsert,
            kResetGuard, kBulkLoad, kReplaceRange, kMergeCone, kMergeMaterialize, kN };
    inline const char* name(int p) {
        static const char* n[] = { "REBUILD (whole fn)", "apply_refit", "dump_sorted", "line_cover",
                                "piece_split", "middle_split", "merge_flanks(rebuild)",
                                "merge_flanks(insert)", "store_insert",
                                "  .reset_guard(exact_dev)", "  .bulk_load",
                                "  .replace_range(tree)", "  .merge_cone(O(w_m))",
                                "  .merge_materialise" };
        return n[p];
    }

    inline std::vector<double> samples[kN];
    inline double last[kN];
    inline void clear_last() { for (int i = 0; i < kN; ++i) last[i] = 0.0; }
    struct Scope {
        int p; std::chrono::steady_clock::time_point t0;
        explicit Scope(int p_) : p(p_), t0(std::chrono::steady_clock::now()) {}
        ~Scope() {
            const double ns =
                std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
            samples[p].push_back(ns);
            last[p] += ns;
        }
    };

    inline void reset() { for (int i = 0; i < kN; ++i) samples[i].clear(); clear_last(); }
    }

    #define CEDAR_PHASE(p) ::li::phase::Scope _li_ps_##__LINE__(::li::phase::p)
    #define CEDAR_PHASE_OP_BEGIN() ::li::phase::clear_last()
#else
    #define CEDAR_PHASE(p) ((void)0)
    #define CEDAR_PHASE_OP_BEGIN() ((void)0)
#endif

// uses memmove insert, so O(w_s) per op
using SegBlock = detail::SegmentStore;
struct Segment {
    Key key_low;
    std::size_t count;
    LinearModel model;
    SegBlock store;
    detail::DriftGuard guard;
    int refit_budget;
    // probe for potential merigng of adjacent segments every n mutations
    std::uint32_t muts_since_probe = 0;
    // TODO: concurrency primitive
    std::uint64_t version;

    std::size_t size() const { return count; }
};

class CedarIndex {
public:
    // model fits to epsilon * kDefaultFitRatio for extra some headroom
    static constexpr double kDefaultFitRatio = 0.75;
    CedarIndex(double epsilon,
               std::size_t w_s = 2048,
               std::size_t w_m = 1024,
               int b = 11,
               double fit_ratio = kDefaultFitRatio)
        : epsilon_(epsilon),
          epsilon_fit_(fit_ratio * epsilon),
          w_s_(w_s),
          w_m_(w_m),
          b_(b),
          max_capacity_slots_(SegBlock::capacity_for_key_cap(w_s)) {
        CEDAR_ASSERT(fit_ratio > 0.0 && fit_ratio <= 1.0);
        CEDAR_CHECK(w_s >= w_m,
                 "CedarIndex: w_s (%zu) < w_m (%zu) is INVALID -- merges install up to w_m keys "
                 "into blocks whose ceiling derives from w_s",
                 w_s, w_m);
    }

    void build(const std::vector<Key>& keys) {
        // CEDAR_MAPPING_VECTOR is really just a correctness check
        // TODO: Remove this compiler flag when finished
        #if defined(CEDAR_MAPPING_VECTOR)
                mapping_table_.clear();
        #else
                table_.clear();
        #endif
        if (keys.empty()) return;

        std::vector<detail::FittedSegment> specs = detail::segment_stream(keys, epsilon_fit_);

        for (const auto& spec : specs) {
            std::size_t offset = 0;

            while (offset < spec.count) {
                std::size_t take = std::min(spec.count - offset, w_s_);
                std::span<const Key> chunk_keys(keys.data() + spec.base_rank + offset, take);
                std::vector<Payload> payloads(take);

                for (std::size_t j = 0; j < take; ++j)
                    payloads[j] = Payload(spec.base_rank + offset + j);

                if (take == spec.count) {
                    push_back_segment(make_segment_with_model(chunk_keys, payloads, spec.model));
                } else {
                    push_back_segment(make_segment(chunk_keys, payloads));
                }

                offset += take;
            }
        }
    }

    Status insert(Key key, Payload payload) {
        CEDAR_PHASE_OP_BEGIN();
        if (table_size() == 0) {
            std::vector<Key> one_key{ key };
            std::vector<Payload> one_payload{ payload };
            push_back_segment(make_segment(one_key, one_payload));
            return Status::ok;
        }

        const auto loc = route_located(key);
        const std::size_t index = loc.index;
        Segment& segment = *loc.seg;

        CEDAR_ASSERT(segment.store.find(key) == std::nullopt);

        bool front_insert = key < segment.key_low;
        SegBlock::EditResult edit;

        {
            CEDAR_PHASE(kStoreInsert);
            if (segment.store.empty() || key > segment.store.max_key()) {
                edit = segment.store.append(key, payload);
            } else {
                edit = segment.store.insert(key, payload);
            }
        }

        segment.count += 1;
        segment.muts_since_probe += 1;

        bool in_band;

        if (front_insert) {
            // front insert is absorbed by intercept
            double beta_rebased = segment.model.beta + 1.0 - segment.model.alpha * double(segment.key_low - key);
            in_band = segment.guard.try_absorb_insert(beta_rebased, epsilon_, false);
            
            if (in_band) {
                segment.model.beta = beta_rebased;
                segment.key_low = key;
            #if !defined(CEDAR_MAPPING_VECTOR)
                // only front inserts change key_low
                // key_low is the seg tree's separator for routing
                // since the seg tree we use is a copy, need to update the key_low
                table_.set_sep(index, key);
            #endif
            }
        } else {
            Rank arriving_rank = segment.store.local_rank(edit.slot);
            double arriving_deviation = line_at(segment.model, key, segment.key_low) - double(arriving_rank);
            bool later_keys_shifted = arriving_rank + 1 < segment.count;
            in_band = segment.guard.try_absorb_insert(arriving_deviation, epsilon_, later_keys_shifted);
        }

        if (in_band) {
            if (segment.count <= w_s_) {
                segment.version += 1;
                try_merge_flanks(index, index + 1);
                return Status::ok;
            }
            const std::size_t k = std::size_t(apply_middle_split(index));
            try_merge_flanks(index, index + k);

            return Status::ok;
        }

        rebuild_broken_segment(index);

        return Status::ok;
    }

    Status erase(Key key) {
        if (table_size() == 0) return Status::not_found;

        const auto loc = route_located(key);
        const std::size_t index = loc.index;
        Segment& segment = *loc.seg;

        if (key < segment.key_low) return Status::not_found;

        std::uint64_t count_before = segment.count;
        SegBlock::EditResult edit = segment.store.erase(key);

        if (!edit.found) return Status::not_found;

        Rank removed_rank = edit.slot;
        segment.count -= 1;
        segment.muts_since_probe += 1;

        if (segment.count == 0) {
            remove_segment(index);
            if (index > 0 && index < table_size())
                try_merge_pair(index - 1);
            return Status::ok;
        }

        bool later_keys_shifted = removed_rank + 1 < count_before;
        bool in_band = segment.guard.try_absorb_erase(epsilon_, later_keys_shifted);

        if (!in_band) {
            rebuild_broken_segment(index);
        } else {
            segment.version += 1;
            try_merge_flanks(index, index + 1);
        }

        return Status::ok;
    }

    Result<Payload> lookup_in_segment(const Segment& segment, Key key) const {
        if (key < segment.key_low) return Status::not_found;
        if (segment.store.empty()) return Status::not_found;

        const auto [lo_rank, hi_rank] = local_window(segment, key);
        std::size_t lo_slot = segment.store.slot_of_rank(lo_rank);
        std::size_t hi_slot = segment.store.slot_of_rank(hi_rank);
        std::optional<std::size_t> hit = segment.store.find_in(key, lo_slot, hi_slot);

        if (!hit) return Status::not_found;

        return segment.store.payload_at(*hit);
    }

    // linear scan is faster
    Result<Payload> point_lookup(Key key) const {
        if (table_size() == 0) return Status::not_found;

        const auto loc = route_located(key);

        return lookup_in_segment(*loc.seg, key);
    }

    #if defined(CEDAR_H0_DIAG)
    void diagnose_missing(Key key) const {
        if (table_size() == 0) {
            std::fprintf(stderr, "H0DIAG key=%llu: EMPTY TABLE\n", (unsigned long long)key);
            return;
        }
        const auto loc = route_located(key);
        const std::size_t index = loc.index;
        const Segment& s = *loc.seg;
        const double pred = line_at(s.model, key, s.key_low);
        const auto win = local_window(s, key);
        const std::optional<std::size_t> in_routed = s.store.find(key);
        std::fprintf(stderr,
            "H0DIAG key=%llu routed=%zu/%zu key_low=%llu count=%zu pred=%.1f "
            "window=[%llu,%llu] guard=[%.2f,%.2f] in_routed=%s",
            (unsigned long long)key, index, table_size(),
            (unsigned long long)s.key_low, s.count, pred,
            (unsigned long long)win.first, (unsigned long long)win.second,
            s.guard.deviation_floor(), s.guard.deviation_ceiling(),
            in_routed ? "YES" : "no");
        if (key < s.key_low) {
            std::fprintf(stderr, "  CLASS=SEPARATOR\n");
        } else if (in_routed) {
            std::fprintf(stderr, "  actual_rank=%llu  CLASS=WINDOW_MISS\n", (unsigned long long)s.store.local_rank(*in_routed));
            return;
        } else {
            std::fprintf(stderr, "\n");
        }
        for (std::size_t i = 0; i < table_size(); ++i) {
            if (i == index) continue;
            if (seg(i).store.find(key)) {
                std::fprintf(stderr,
                    "H0DIAG key=%llu FOUND in segment %zu (routed %zu, delta %+lld) "
                    "key_low=%llu count=%zu  CLASS=MISROUTED\n",
                    (unsigned long long)key, i, index,
                    (long long)i - (long long)index,
                    (unsigned long long)seg(i).key_low, seg(i).count);
                return;
            }
        }
        if (!in_routed)
            std::fprintf(stderr, "H0DIAG key=%llu CLASS=DROPPED (in no segment's block)\n", (unsigned long long)key);
    }

    #endif

    // measures how far the index currently is from exact W irreducibility
    // phi (combined count <= w_m) : the pair would merge if probed, but not cause lazy
    // gate coverable = combined count > w_m, means tis is coverable, but w-m doesnt allow merge
    struct StaleCensus {
        std::size_t pairs = 0,
        phi = 0,
        gated_cov = 0;
    };

    StaleCensus stale_pair_census() const {
        StaleCensus c;

        if (table_size() < 2) return c;

        for (std::size_t i = 0; i + 1 < table_size(); ++i) {
            const Segment& left = seg(i);
            const Segment& right = seg(i + 1);
            c.pairs += 1;

            if (left.store.empty()) continue;

            detail::StreamingCone cone(left.store.keys_view()[0], epsilon_fit_);
            std::uint64_t r = 0;
            bool cov = true;

            for (Key k : left.store.keys_view())
                if (!cone.try_extend(k, r++)) {
                    cov = false;
                    break;
                }
            if (cov)
                for (Key k : right.store.keys_view())
                    if (!cone.try_extend(k, r++)) {
                        cov = false;
                        break;
                    }

            if (cov) {
                if (left.count + right.count > w_m_) c.gated_cov += 1;
                else c.phi += 1;
            }
        }

        return c;
    }

    // streaming scan, visit is the function to call once you are at a valid key. see range_lookup on how to use
    template <class F>
    void range_scan(Key lo, Key hi, F&& visit) const {
        if (hi < lo || table_size() == 0) return;

        for (std::size_t index = route(lo); index < table_size(); ++index) {
            const Segment& segment = seg(index);

            if (segment.key_low > hi) break;

            for (Key k : segment.store.keys_view()) {
                if (k < lo) continue;
                if (k > hi) return;
                visit(k);
            }
        }
    }

    std::vector<Key> range_lookup(Key lo, Key hi) const {
        std::vector<Key> out;
        range_scan(lo, hi, [&](Key k) { out.push_back(k); });

        return out;
    }

    // only used for measuring how much space the index takes
    struct SegStats {
        std::size_t M = 0;
        double slots = 0;
        double keys = 0;
    };

    SegStats segment_stats() const {
        SegStats s;
        s.M = table_size();
        #if defined(CEDAR_MAPPING_VECTOR)
            for (const auto& p : mapping_table_) {
        #else
            auto body = [&](const std::unique_ptr<Segment>& p) {
        #endif
        s.slots += double(p->store.capacity());
        s.keys += double(p->store.size());
        #if defined(CEDAR_MAPPING_VECTOR)
            }
        #else
            };
            table_.for_each_item(body);
        #endif
        return s;
    }

    // isolating it here for testing
    std::size_t route_for_test(Key key) const { return route(key); }
    std::size_t table_size_for_test() const { return table_size(); }

    template <class F>
    void for_each_segment_for_test(F&& f) const {
        #if defined(CEDAR_MAPPING_VECTOR)
            for (const auto& p : mapping_table_) f(*p);
        #else
            table_.for_each_item([&](const std::unique_ptr<Segment>& p) { f(*p); });
        #endif
    }

    void set_merge_delta(std::size_t d) { merge_delta_ = d; }
    void set_merge_prop(std::size_t k) { merge_prop_ = k; }

    std::uint64_t merge_skips() const { return merge_skips_; }
    std::uint64_t cover_recomputes() const { return cover_recomputes_; }
    std::uint64_t merge_probes() const { return merge_probes_; }
    std::uint64_t merge_probe_hits() const { return merge_probe_hits_; }
    std::uint64_t piece_splits() const { return piece_splits_; }
    std::uint64_t force_splits() const { return force_splits_; }

    // track using --margins
    // per segment birth, headroom = absorbs before tripping
    enum class InstallSite : std::uint8_t { kBuild = 0, kRefit, kSplitHalf, kSplitPiece, kMerge, kCount };
    static const char* site_name(InstallSite s) {
        switch (s) {
            case InstallSite::kBuild: return "build";
            case InstallSite::kRefit: return "refit";
            case InstallSite::kSplitHalf: return "split_half";
            case InstallSite::kSplitPiece: return "split_piece";
            case InstallSite::kMerge: return "merge";
            default: return "?";
        }
    }

    struct MarginRecord {
        float headroom;
        float cone;
        std::uint32_t count;
        InstallSite site;
    };

    const std::vector<MarginRecord>& margin_log() const { return margin_log_; }
    void set_margin_logging(bool on) { log_margins_ = on; }
    void clear_margin_log() { margin_log_.clear(); }
    // TODO: see if I still need this ^
    // =====================

    double epsilon() const { return epsilon_; }
    double epsilon_fit() const { return epsilon_fit_; }

    std::size_t w_s() const { return w_s_; }
    std::size_t w_m() const { return w_m_; }


private:
    // LI MAPPING VECTOR is the same op stream with identical stuff, but just simpler
    // helps me track if the mapping table B+tree is good
    #if defined(CEDAR_MAPPING_VECTOR)
        std::vector<std::unique_ptr<Segment>> mapping_table_;
        Segment& seg(std::size_t i) { return *mapping_table_[i]; }
        const Segment& seg(std::size_t i) const { return *mapping_table_[i]; }
        std::size_t table_size() const { return mapping_table_.size(); }
    #else
        // The mapping table is a B+tree of segment descriptors at B=32.
        // Routing and structual ops are O(log M) with cache resident hops.
        // why didnt i use learned/computing routing or some type of RMI based technique?
        // i didnt like the cascading model rebuild, also big repair path. this caused bad latency. but maybe a TODO in the future to better optimize this instead of using a B+tree here
        // TODO sweep B
        detail::SegmentTree<std::unique_ptr<Segment>, 32> table_;
        Segment& seg(std::size_t i) {
            return *table_.at(i);
        }
        const Segment& seg(std::size_t i) const {
            return *table_.at(i);
        }
        std::size_t table_size() const {
            return table_.size();
        }
    #endif

    // Index + pointer from a single descent: every op path needs both (pointer to edit the segment,
    // index for neighbors/installs)
    // resolve them both at the same time to negate extra work
    struct Located {
        std::size_t index;
        Segment* seg;
    };

    Located route_located(Key key) const {
        #if defined(CEDAR_MAPPING_VECTOR)
            const std::size_t i = route(key);
            return { i, const_cast<Segment*>(&seg(i)) };
        #else
            const auto loc = table_.route_located(key);
            return { loc.index, loc.item ? loc.item->get() : nullptr };
        #endif
    }

    // segments i and i+1 used in
    struct SegPair { Segment* a; Segment* b; };
    SegPair at_pair(std::size_t i) {
        #if defined(CEDAR_MAPPING_VECTOR)
            return { &seg(i), &seg(i + 1) };
        #else
            const auto p = table_.at_pair(i);
            return { p.a->get(), p.b->get() };
        #endif
    }

    // only really used in build()
    void push_back_segment(Segment&& sgm) {
        #if defined(CEDAR_MAPPING_VECTOR)
            mapping_table_.push_back(std::make_unique<Segment>(std::move(sgm)));
        #else
            const Key sep = sgm.key_low;
            table_.insert_at(table_.size(), sep, std::make_unique<Segment>(std::move(sgm)));
        #endif
    }

    mutable std::uint64_t cover_recomputes_ = 0;
    mutable std::uint64_t merge_probes_ = 0;
    mutable std::uint64_t merge_probe_hits_ = 0;
    mutable std::uint64_t merge_skips_ = 0;
    mutable std::uint64_t piece_splits_ = 0;
    mutable std::uint64_t force_splits_ = 0;

    mutable std::vector<MarginRecord> margin_log_;

    bool in_rebuild_ = false;
    mutable double refit_cone_margin_ = -1.0;
    mutable double merge_cone_margin_ = -1.0;

    bool log_margins_ = false;

    double epsilon_;
    // eps_fit: what cones are fit to. margin = eps_inv - eps_fit.
    double epsilon_fit_;
    std::size_t w_s_;
    std::size_t w_m_;
    std::size_t merge_delta_ = 0;

    // this is the proportional merge throttle, a pair is re probed for coverability only after its contents have moved by
    // pair_count/32. default 32 seems to work best
    // TODO: better grid search this default
    std::size_t merge_prop_ = 32;
    int b_;
    std::size_t max_capacity_slots_;
    std::size_t route(Key key) const {
        #if defined(CEDAR_MAPPING_VECTOR)
            auto it = std::upper_bound(mapping_table_.begin(), mapping_table_.end(), key,
                [](Key k, const std::unique_ptr<Segment>& s) { return k < s->key_low; });
            if (it == mapping_table_.begin()) return 0;
            return std::size_t((it - mapping_table_.begin()) - 1);
        #else
            return table_.route(key);
        #endif
    }

    // this is the window of the last mile scan, error guaranteed <= epsilon
    std::pair<Rank, Rank> local_window(const Segment& segment, Key key) const {
        Rank pred = predict(segment.model, key, segment.key_low);
        Rank max_rank = Rank(segment.count - 1);

        if (pred > max_rank) pred = max_rank;

        const double ce = std::min(segment.guard.deviation_ceiling(), epsilon_);
        const double fl = std::max(segment.guard.deviation_floor(), -epsilon_);
        const std::int64_t down = std::int64_t(std::ceil(ce)) + 1;
        const std::int64_t up = std::int64_t(std::ceil(-fl)) + 1;
        const std::int64_t lo = std::int64_t(pred) - (down > 0 ? down : 0);
        const std::int64_t hi = std::int64_t(pred) + (up > 0 ? up : 0) + 1;

        Rank lo_rank = lo > 0 ? Rank(lo) : Rank{ 0 };
        Rank hi_rank = Rank(hi);

        if (hi_rank > max_rank) hi_rank = max_rank;

        return { lo_rank, hi_rank };
    }

    Segment make_segment_with_model(std::span<const Key> keys,
                                    std::span<const Payload> payloads,
                                    LinearModel model,
                                    InstallSite site = InstallSite::kBuild,
                                    double cone_margin = -1.0) {
        Segment segment;
        segment.key_low = keys[0];
        segment.count = keys.size();
        segment.model = model;

        {
            CEDAR_PHASE(kBulkLoad);
            segment.store = SegBlock::bulk_load(keys, payloads, max_capacity_slots_);
        }

        segment.muts_since_probe = 0;
        reset_guard(segment, site, cone_margin);
        segment.refit_budget = b_;
        segment.version = 0;

        return segment;
    }

    Segment make_segment_piece(std::span<const Key> keys,
                               std::span<const Payload> payloads,
                               LinearModel model) {
        return make_segment_with_model(keys, payloads, model, InstallSite::kSplitPiece);
    }

    std::size_t append_cover_segments(std::span<const Key> keys,
                                      std::span<const Payload> payloads,
                                      std::vector<Segment>& out) {
        detail::LineCoverResult verdict = detail::minimal_line_cover(keys, epsilon_fit_);

        if (verdict.status == detail::LineCoverStatus::COVERABLE) {
            out.push_back(make_segment_with_model(keys, payloads, verdict.model,InstallSite::kSplitHalf));
            return 1;
        }

        for (const auto& piece : verdict.pieces) {
            const std::size_t len = piece.end - piece.begin;
            out.push_back(make_segment_with_model(keys.subspan(piece.begin, len),
                                                  payloads.subspan(piece.begin, len),
                                                  piece.model, InstallSite::kSplitPiece));
        }

        return verdict.pieces.size();
    }

    // view into block
    struct SortedView {
        std::span<const Key> keys;
        std::span<const Payload> payloads;
    };

    static SortedView sorted_view(const detail::SegmentStore& block) {
        return { block.keys_view(), block.payloads_view() };
    }

    Segment make_segment(std::span<const Key> keys,
                         std::span<const Payload> payloads,
                         InstallSite site = InstallSite::kBuild) {
        detail::LineCoverResult verdict = detail::minimal_line_cover(keys, epsilon_fit_);

        CEDAR_CHECK(verdict.status == detail::LineCoverStatus::COVERABLE,
                 "make_segment: %zu keys not eps_fit-coverable (install site %d) -- caller broke "
                 "the coverability contract",
                 keys.size(), int(site));

        return make_segment_with_model(keys, payloads, verdict.model, site);
    }

    // The exact extreme deviations of a segment's live keys against its current model
    std::pair<double, double> exact_deviations(const Segment& segment) const {
        double lo = std::numeric_limits<double>::infinity();
        double hi = -std::numeric_limits<double>::infinity();
        std::size_t rank = 0;

        for (Key k : segment.store.keys_view()) {
            double d = line_at(segment.model, k, segment.key_low) - double(rank++);
            lo = std::min(lo, d);
            hi = std::max(hi, d);
        }

        if (lo > hi) {
            lo = 0.0;
            hi = 0.0;
        }

        return { lo, hi };
    }

    // Measure the new model's min/max error, start the guard from it, fail is exceeds eps
    void reset_guard(Segment& segment, InstallSite site = InstallSite::kBuild, double cone_margin = -1.0) {
        CEDAR_PHASE(kResetGuard);
        auto [lo, hi] = exact_deviations(segment);
        segment.guard.reset(lo, hi);

        if (log_margins_) {
            margin_log_.push_back(
                MarginRecord{float(epsilon_ + lo), float(cone_margin), std::uint32_t(segment.count), site}
            );
        }

        // floating point error for h0 check
        const double h0_slack = 16.0 * double(segment.count + 1) * std::numeric_limits<double>::epsilon();
        CEDAR_CHECK(segment.guard.holds(epsilon_ + h0_slack),
                 "H0 AT BIRTH violated: install site %d, count %zu, exact deviations [%.3f, %.3f], "
                 "eps_inv %.1f -- an install path produced a non-covering model",
                 int(site), segment.count, lo, hi, epsilon_);
    }

    // Guard tripped: fit a new model for this segment, or split it (budget spent, over cap,
    // or not fittable), then check if the new neighbors can merge.
    void rebuild_broken_segment(std::size_t index) {
        cover_recomputes_ += 1;
        CEDAR_PHASE(kRebuildTotal);
        // raii trick so i dont have to update state everywhere
        // also only used for measuring
        struct InRebuild {
            bool& f;
            explicit InRebuild(bool& x) : f(x) {
                f = true;
            }

            ~InRebuild() {
                f = false;
            }
        }
        _li_ir(in_rebuild_);

        SortedView view{};

        {
            CEDAR_PHASE(kDumpSorted);
            view = sorted_view(seg(index).store);
        }

        const std::span<const Key> keys = view.keys;
        const std::span<const Payload> payloads = view.payloads;

        CEDAR_ASSERT(!keys.empty());

        const Key live_key_low = keys[0];
        detail::LineCoverResult verdict;

        {
            CEDAR_PHASE(kLineCover);
            verdict = detail::minimal_line_cover(keys, epsilon_fit_);
        }

        const bool coverable = (verdict.status == detail::LineCoverStatus::COVERABLE);
        refit_cone_margin_ = -1.0;

        if (log_margins_ && coverable) {
            detail::StreamingCone mc(live_key_low, epsilon_fit_);
            for (std::size_t i2 = 0; i2 < keys.size(); ++i2) mc.try_extend(keys[i2], i2);
            refit_cone_margin_ = mc.current_margin();
        }

        if (coverable) {
            const LinearModel streamed_model = verdict.model;
            bool force_split = seg(index).refit_budget <= 0 && seg(index).count > w_m_;

            if (force_split) force_splits_ += 1;

            if (!force_split) {
                apply_refit(index, streamed_model, live_key_low);

                if (seg(index).count > w_s_) {
                    const std::size_t k = std::size_t(apply_middle_split(index));
                    try_merge_flanks(index, index + k);
                    return;
                }

                try_merge_flanks(index, index + 1);
                return;
            }

            const std::size_t kf = std::size_t(apply_middle_split(index));
            try_merge_flanks(index, index + kf);
            return;
        }

        piece_splits_ += 1;
        const std::size_t k = apply_piece_split_timed(index, verdict.pieces, keys, payloads);
        try_merge_flanks(index, index + k);
    }

    void apply_refit(std::size_t index, LinearModel new_model, Key new_key_low) {
        CEDAR_PHASE(kApplyRefit);
        Segment& segment = seg(index);
        segment.key_low = new_key_low;
        segment.model = new_model;

        #if !defined(CEDAR_MAPPING_VECTOR)
            table_.set_sep(index, new_key_low);
        #endif

        reset_guard(segment, InstallSite::kRefit, refit_cone_margin_);

        segment.refit_budget -= 1;
        segment.version += 1;
    }

    // split segment in half and refit each
    int apply_middle_split(std::size_t index) {
        CEDAR_PHASE(kMiddleSplit);
        const SortedView view = sorted_view(seg(index).store);

        CEDAR_ASSERT(view.keys.size() >= 2);

        std::size_t mid = view.keys.size() / 2;
        std::vector<Segment> parts;
        std::size_t k = 0;

        k += append_cover_segments(view.keys.subspan(0, mid), view.payloads.subspan(0, mid), parts);
        k += append_cover_segments(view.keys.subspan(mid), view.payloads.subspan(mid), parts);

        {
            CEDAR_PHASE(kReplaceRange);
            replace_range(index, 1, std::move(parts));
        }

        return int(k);
    }

    std::size_t apply_piece_split_timed(std::size_t index,
                                        const std::vector<detail::LineCoverPiece>& pieces,
                                        std::span<const Key> keys,
                                        std::span<const Payload> payloads) {
        CEDAR_PHASE(kPieceSplit);
        return apply_piece_split(index, pieces, keys, payloads);
    }

    // Takes pieces and applies splits in mapping table. **No refitting here**
    std::size_t apply_piece_split(std::size_t index,
                                  const std::vector<detail::LineCoverPiece>& pieces,
                                  std::span<const Key> keys,
                                  std::span<const Payload> payloads) {
        std::vector<Segment> parts;
        parts.reserve(pieces.size());

        for (const auto& piece : pieces) {
            std::size_t len = piece.end - piece.begin;
            parts.push_back(
                make_segment_piece(
                    std::span<const Key>(keys.data() + piece.begin, len),
                    std::span<const Payload>(payloads.data() + piece.begin, len),
                    piece.model
                )
            );
        }

        std::size_t k = parts.size();

        {
            CEDAR_PHASE(kReplaceRange);
            replace_range(index, 1, std::move(parts));
        }

        return k;
    }

    // only left and right are provably mergable after split
    // irreducibility lemma
    void try_merge_flanks(std::size_t lo, std::size_t hi) {
        #if defined(CEDAR_PHASE_TIMING)
            ::li::phase::Scope _li_mf(in_rebuild_ ? ::li::phase::kMergeFlanksInRebuild
                                                    : ::li::phase::kMergeFlanksInInsert);
        #endif
        if (hi < table_size())
            try_merge_pair(hi - 1);

        if (lo > 0)
            try_merge_pair(lo - 1);
    }

    bool try_merge_pair(std::size_t left_index) {
        if (left_index + 1 >= table_size()) return false;

        const auto pr = at_pair(left_index);
        Segment& left = *pr.a;
        Segment& right = *pr.b;

        if (left.count + right.count > w_m_) return false;

        if (merge_delta_ > 0 || merge_prop_ > 0) {
            std::size_t threshold = merge_delta_;

            if (merge_prop_ > 0) {
                const std::size_t prop = (left.count + right.count) / merge_prop_;
                if (prop > threshold) threshold = prop;
            }

            if (threshold > 0) {
                const std::uint64_t moved =
                    std::uint64_t(left.muts_since_probe) + std::uint64_t(right.muts_since_probe);

                if (moved < threshold) {
                    merge_skips_ += 1;
                    return false;
                }

                left.muts_since_probe = 0;
                right.muts_since_probe = 0;
            }
        }

        #ifdef CEDAR_NO_MERGE
            return false;
        #endif

        merge_probes_ += 1;

        CEDAR_ASSERT(!left.store.empty() && !right.store.empty());

        const Key merged_key_low = left.store.keys_view()[0];
        // prefilter to help out p99, if infeasible on subset of points, then it's infeasible on all
        // here, prereject merges in O(1) before the full stream
        // Measured 64M keys: over 50% streams killed; wm=1024 sorted inserts p99 10.3k -> 220ns, same with big wins on delete p99
        // doesnt help much on wm=256 though
        // TODO: this releases the wm bounding merge min size for too much merging.
        //     test out how raising wm now can have better pareto frontier for space + p99 latency?
        {
            detail::StreamingCone pre(merged_key_low, epsilon_fit_);
            const std::size_t lc = left.count;
            const std::size_t total = left.count + right.count;

            bool feasible = pre.try_extend(merged_key_low, 0);

            if (feasible && lc > 1)
                feasible = pre.try_extend(left.store.max_key(), lc - 1);
            if (feasible)
                feasible = pre.try_extend(right.store.keys_view()[0], lc);
            if (feasible && total - 1 > lc)
                feasible = pre.try_extend(right.store.max_key(), total - 1);

            if (!feasible) return false;
        }

        LinearModel _cone_model;

        {
            CEDAR_PHASE(kMergeCone);
            detail::StreamingCone cone(merged_key_low, epsilon_fit_);
            uint64_t local = 0;

            for (Key k : left.store.keys_view()) {
                if (!cone.try_extend(k, local++)) return false;
            }

            for (Key k : right.store.keys_view()) {
                if (!cone.try_extend(k, local++)) return false;
            }

            merge_cone_margin_ = cone.current_margin();
            _cone_model = cone.finalize().model;
        }

        const LinearModel model = _cone_model;

        CEDAR_PHASE(kMergeMaterialize);
        Segment merged = make_merged_segment(left, right, merged_key_low, model);
        replace_one(left_index, 2, std::move(merged));
        merge_probe_hits_ += 1;

        return true;
    }

    Segment make_merged_segment(Segment& left, Segment& right, Key merged_key_low, LinearModel model) {
        Segment merged;
        {
            const detail::SegmentStore& lb = left.store;
            const detail::SegmentStore& rb = right.store;

            merged.key_low = merged_key_low;
            merged.model = model;

            {
                CEDAR_PHASE(kBulkLoad);
                merged.store = detail::SegmentStore::bulk_load2(lb.keys_view(), lb.payloads_view(),
                                                                rb.keys_view(), rb.payloads_view(),
                                                                max_capacity_slots_);
            }

            merged.count = merged.store.size();
            merged.muts_since_probe = 0;

            reset_guard(merged, InstallSite::kMerge, merge_cone_margin_);

            merged.refit_budget = b_;
            merged.version = 0;
        }

        return merged;
    }

    void remove_segment(std::size_t index) {
        #if defined(CEDAR_MAPPING_VECTOR)
            mapping_table_.erase(mapping_table_.begin() + std::ptrdiff_t(index));
        #else
            table_.erase_at(index);
        #endif
    }

    void replace_range(std::size_t first, std::size_t remove_count, std::vector<Segment>&& news) {
        #if defined(CEDAR_MAPPING_VECTOR)
            std::vector<std::unique_ptr<Segment>> boxed;
            boxed.reserve(news.size());
            for (auto& n : news) boxed.push_back(std::make_unique<Segment>(std::move(n)));
            auto at = std::ptrdiff_t(first);
            auto rm = std::ptrdiff_t(remove_count);
            mapping_table_.erase(mapping_table_.begin() + at, mapping_table_.begin() + at + rm);
            mapping_table_.insert(mapping_table_.begin() + at,
                                std::make_move_iterator(boxed.begin()),
                                std::make_move_iterator(boxed.end()));
        #else
            for (std::size_t k = 0; k < remove_count; ++k) table_.erase_at(first);
            for (std::size_t k = 0; k < news.size(); ++k) {
                const Key sep = news[k].key_low;
                table_.insert_at(first + k, sep, std::make_unique<Segment>(std::move(news[k])));
            }
        #endif
    }

    void replace_one(std::size_t first, std::size_t remove_count, Segment&& one) {
        #if defined(CEDAR_MAPPING_VECTOR)
            auto at = std::ptrdiff_t(first);
            auto rm = std::ptrdiff_t(remove_count);
            mapping_table_.erase(mapping_table_.begin() + at, mapping_table_.begin() + at + rm);
            mapping_table_.insert(mapping_table_.begin() + at, std::make_unique<Segment>(std::move(one)));
        #else
            for (std::size_t k = 0; k < remove_count; ++k) table_.erase_at(first);
            const Key sep = one.key_low;
            table_.insert_at(first, sep, std::make_unique<Segment>(std::move(one)));
        #endif
    }
};

}