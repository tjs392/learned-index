#include "li/cedar_index.hpp"
#include <chrono>
#include <vector>
#include <algorithm>
#include <random>
#include <cstdio>
using namespace li; using ns=std::chrono::nanoseconds;
static double pct(std::vector<double>&v,double p){std::sort(v.begin(),v.end());return v[(size_t)(p*(v.size()-1))];}
int main(){
    const size_t N=200000;
    std::vector<Key> base; for(Key k=1;k<=N;++k) base.push_back(k);
    
    std::mt19937_64 g(7); std::vector<Key> perm=base; std::shuffle(perm.begin(),perm.end(),g);
    std::vector<Key> bulk(perm.begin(), perm.begin()+N/2);
    std::vector<Key> rest(perm.begin()+N/2, perm.end());
    std::sort(bulk.begin(), bulk.end());

    auto run=[&](const char* name, std::vector<Key> stream){
        CedarIndex idx(32.0, 256, 128, 8); idx.build(bulk);
        std::vector<double> lat; lat.reserve(stream.size());
        for(Key k: stream){
            auto t0=std::chrono::steady_clock::now();
            idx.insert(k,(Payload)k);
            auto t1=std::chrono::steady_clock::now();
            lat.push_back((double)std::chrono::duration_cast<ns>(t1-t0).count());
        }
        std::printf("%-12s  M(build)=%5zu M(end)=%5zu  p50=%6.0f  p99.9=%8.0f  p99.99=%8.0f  max=%9.0f  recomputes=%zu\n",
            name, (size_t)0, idx.num_segments(), pct(lat,0.5), pct(lat,0.999), pct(lat,0.9999), pct(lat,1.0), idx.cover_recomputes());
    };
    std::vector<Key> s_asc=rest; std::sort(s_asc.begin(),s_asc.end());
    std::vector<Key> s_desc=rest; std::sort(s_desc.rbegin(),s_desc.rend());
    std::vector<Key> s_shuf=rest;
    std::printf("=== CEDAR insert-tail (ns) by order, uniform-sample bulk=%zu (M=%zu), insert=%zu ===\n",
                N/2, (size_t)0, rest.size());
    { CedarIndex t(32.0); t.build(bulk); std::printf("(bulk segments M = %zu)\n", t.num_segments()); }
    run("sorted-asc", s_asc);
    run("shuffled",   s_shuf);
    run("adversarial",s_desc);
    return 0;
}