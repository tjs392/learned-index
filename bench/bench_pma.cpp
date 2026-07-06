#include "li/pma.hpp"
#include <chrono>
#include <random>
#include <vector>
#include <span>
#include <cstdio>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <cmath>

using li::detail::PmaBlock;
using li::Key;

static std::uint64_t g_moves = 0;

struct Counted {
    std::uint64_t v = 0;
    Counted() = default;
    Counted(std::uint64_t x) : v(x) {}
    Counted(const Counted&) = default;
    Counted(Counted&&) = default;
    Counted& operator=(const Counted& o) { ++g_moves; v = o.v; return *this; }
    Counted& operator=(Counted&& o) { ++g_moves; v = o.v; return *this; }
};

enum class Pattern { Random, Hammer, Append };

static std::vector<Key> order_for(Pattern p, std::size_t n, std::mt19937_64& rng) {
    std::vector<Key> order(n);
    std::iota(order.begin(), order.end(), Key(1));
    if (p == Pattern::Random) std::shuffle(order.begin(), order.end(), rng);
    else if (p == Pattern::Hammer) std::reverse(order.begin(), order.end());
    return order;
}

static std::uint64_t measure_moves(Pattern p, std::size_t n, std::mt19937_64& rng) {
    std::vector<Key> order = order_for(p, n, rng);
    PmaBlock<Counted> b =
        PmaBlock<Counted>::bulk_load(std::span<const Key>{}, std::span<const Counted>{});
    g_moves = 0;
    if (p == Pattern::Append) for (Key k : order) b.append(k, Counted{k});
    else                      for (Key k : order) b.insert(k, Counted{k});
    return g_moves;
}

static double measure_ns(Pattern p, std::size_t n, std::mt19937_64& rng) {
    std::vector<Key> order = order_for(p, n, rng);
    PmaBlock<std::uint64_t> b =
        PmaBlock<std::uint64_t>::bulk_load(std::span<const Key>{}, std::span<const std::uint64_t>{});
    auto t0 = std::chrono::steady_clock::now();
    if (p == Pattern::Append) for (Key k : order) b.append(k, k);
    else                      for (Key k : order) b.insert(k, k);
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / double(n);
}

static void self_check() {
    std::vector<Key> ks(1000);
    std::iota(ks.begin(), ks.end(), Key(1));
    std::vector<Counted> ps;
    for (Key k : ks) ps.push_back(Counted{k});
    g_moves = 0;
    auto b = PmaBlock<Counted>::bulk_load(ks, ps);
    (void)b;
    if (g_moves != ks.size()) {
        std::fprintf(stderr, "counter self-check FAILED: %llu != %zu\n",
                     (unsigned long long)g_moves, ks.size());
        std::exit(1);
    }
    std::printf("counter self-check: bulk_load(%zu) counted %llu placements (exact)\n\n",
                ks.size(), (unsigned long long)g_moves);
}

int main(int argc, char** argv) {
    std::mt19937_64 rng(0xC0FFEEULL);
    self_check();

    std::vector<std::size_t> Ns = {1000, 4000, 16000, 64000, 256000, 1000000};
    if (argc > 1) {
        Ns.clear();
        for (int i = 1; i < argc; ++i) Ns.push_back(std::strtoull(argv[i], nullptr, 10));
    }

    const char* names[] = {"random inserts (expect O(log N))",
                           "hammer/front inserts (expect O(log^2 N))",
                           "append/sequential (expect O(1))"};
    Pattern pats[] = {Pattern::Random, Pattern::Hammer, Pattern::Append};

    for (int pi = 0; pi < 3; ++pi) {
        std::printf("== %s ==\n", names[pi]);
        std::printf("%10s %14s %12s %14s %12s\n",
                    "N", "moves/insert", "/ lgN", "/ (lgN)^2", "ns/insert");
        for (std::size_t N : Ns) {
            std::uint64_t mv = measure_moves(pats[pi], N, rng);
            double ns = measure_ns(pats[pi], N, rng);
            double per = double(mv) / double(N);
            double lg = std::log2(double(N));
            std::printf("%10zu %14.2f %12.3f %14.4f %12.1f\n",
                        N, per, per / lg, per / (lg * lg), ns);
        }
        std::printf("\n");
    }
    return 0;
}