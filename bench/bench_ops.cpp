#include "li/index.hpp"
#include "li/status.hpp"

#include <vector>
#include <set>
#include <random>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <algorithm>

using li::LearnedIndex;
using li::Key;
using li::Payload;

using clk = std::chrono::steady_clock;
static double ns_since(clk::time_point t0) {
    return std::chrono::duration<double, std::nano>(clk::now() - t0).count();
}

static std::vector<Key> gen_dense(uint64_t n) {
    std::vector<Key> k(n);
    for (uint64_t i = 0; i < n; ++i) k[i] = i;
    return k;
}
static std::vector<Key> gen_linear_gap(uint64_t n, std::mt19937_64& rng) {
    std::vector<Key> k; k.reserve(n);
    uint64_t v = 0;
    for (uint64_t i = 0; i < n; ++i) { v += 1 + (rng() % 8); k.push_back(v); }
    return k;
}
static std::vector<Key> gen_clustered(uint64_t n, std::mt19937_64& rng) {
    std::vector<Key> k; k.reserve(n);
    uint64_t v = 0;
    while (k.size() < n) {
        uint64_t run = 200 + (rng() % 800);
        for (uint64_t i = 0; i < run && k.size() < n; ++i) { v += 1 + (rng() % 3); k.push_back(v); }
        v += 100000 + (rng() % 1000000);
    }
    return k;
}
static std::vector<Key> gen_uniform(uint64_t n, std::mt19937_64& rng) {
    std::set<Key> s;
    std::uniform_int_distribution<uint64_t> d(0, ~0ull);
    while (s.size() < n) s.insert(d(rng));
    return std::vector<Key>(s.begin(), s.end());
}

static uint64_t g_sink = 0;

static void bench_read(const char* name, const std::vector<Key>& keys, double eps) {
    LearnedIndex idx(eps);
    auto t0 = clk::now();
    idx.build(keys);
    double build_ms = ns_since(t0) / 1e6;

    const uint64_t M = idx.mapping_table_for_test().size();
    const double avg_w = double(keys.size()) / double(M);

    std::mt19937_64 rng(1);
    const uint64_t Q = keys.size();
    std::vector<Key> probe(Q);
    for (uint64_t i = 0; i < Q; ++i) probe[i] = keys[rng() % keys.size()];

    t0 = clk::now();
    for (uint64_t i = 0; i < Q; ++i) {
        auto r = idx.point_lookup(probe[i]);
        if (r.ok()) g_sink += r.value();
    }
    double look_ns = ns_since(t0) / double(Q);

    std::printf("  %-14s N=%-8llu M=%-8llu avg_w=%-10.1f build=%8.2f ms   lookup=%8.1f ns/op\n",
                name, (unsigned long long)keys.size(), (unsigned long long)M, avg_w, build_ms, look_ns);
}

static void bench_write(const char* name, const std::vector<Key>& base, double eps, uint64_t ops) {
    std::set<Key> present(base.begin(), base.end());
    LearnedIndex idx(eps);
    idx.build(base);

    const uint64_t M = idx.mapping_table_for_test().size();
    const double avg_w = double(base.size()) / double(M);

    std::mt19937_64 rng(2);
    const uint64_t hi = base.empty() ? 1 : (base.back() + 1) * 3 + 10;

    std::vector<Key> to_insert;
    to_insert.reserve(ops);
    while (to_insert.size() < ops) {
        Key k = rng() % hi;
        if (present.insert(k).second) to_insert.push_back(k);
    }

    auto t0 = clk::now();
    for (Key k : to_insert) idx.insert(k, k + 7);
    double ins_ns = ns_since(t0) / double(ops);

    t0 = clk::now();
    for (Key k : to_insert) idx.erase(k);
    double era_ns = ns_since(t0) / double(ops);

    std::printf("  %-14s base=%-7llu M=%-6llu avg_w=%-9.1f insert=%9.1f ns/op   erase=%9.1f ns/op\n",
                name, (unsigned long long)base.size(), (unsigned long long)M, avg_w, ins_ns, era_ns);
}

int main() {
    const double eps = 8.0;
    std::mt19937_64 rng(42);

    const uint64_t NR = 200000;
    std::printf("== build + point lookup (eps=%.1f) ==\n", eps);
    bench_read("dense",      gen_dense(NR),            eps);
    bench_read("linear_gap", gen_linear_gap(NR, rng),  eps);
    bench_read("clustered",  gen_clustered(NR, rng),   eps);
    bench_read("uniform",    gen_uniform(NR, rng),     eps);

    const uint64_t NW = 10000;
    const uint64_t OPS = 2000;
    std::printf("\n== insert + erase (eps=%.1f, %llu ops) ==\n", eps, (unsigned long long)OPS);
    bench_write("dense",      gen_dense(NW),            eps, OPS);
    bench_write("linear_gap", gen_linear_gap(NW, rng),  eps, OPS);
    bench_write("clustered",  gen_clustered(NW, rng),   eps, OPS);
    bench_write("uniform",    gen_uniform(NW, rng),     eps, OPS);

    std::printf("\n(sink=%llu)\n", (unsigned long long)g_sink);
    return 0;
}