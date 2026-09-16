/*
*	Shared coefficient-compression helper for the CUDA engines.
*
*	Both the single-GPU engine (whole mesh) and the multi-GPU engine (one call
*	per x-slab) deduplicate each cell's 12 fp32 update coefficients into two
*	small unique tables plus a per-cell index. This lived twice -- once in
*	engine_cuda.cu, once as build_coeff_slice() in engine_cuda_mgpu.cu -- and
*	the copies drifted: the parallel rewrite only landed on the single-GPU one,
*	leaving the multi-GPU path on the original serial std::string-key version
*	(a heap allocation per cell, once per slab, and untimed). One definition
*	here so the two engines cannot diverge again.
*/

#ifndef ENGINE_CUDA_COEFF_H
#define ENGINE_CUDA_COEFF_H

#include <cuda_runtime.h>
#include <sys/time.h>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <thread>

#include "tools/cuda/check.h"

// Deduplicate each cell's 12-float coefficient set (vv,vi interleaved for the
// voltage kernel; ii,iv for the current kernel, in exactly the order the kernels
// read them) into two small unique-coefficient tables plus a per-cell index, and
// upload all three to the device. Homogeneous regions (free space, uniform mesh)
// collapse to a handful of unique entries, so the tables stay L2-resident and the
// update kernels move a 4-byte index instead of 24 bytes of coefficients per cell.
static inline void openems_build_compressed_coeff(
    const FDTD_FLOAT *vv, const FDTD_FLOAT *vi,
    const FDTD_FLOAT *ii, const FDTD_FLOAT *iv,
    int num_cells,
    void **d_index, bool *index_u16, FDTD_FLOAT **d_vv_vi, FDTD_FLOAT **d_ii_iv,
    unsigned int *num_unique, const char *tag)
{
    // Dedup the 12 fp32 update coeffs per cell into a small unique table + a
    // per-cell index. This is a per-cell hash over 41M+ cells and was the last
    // fully-serial startup cost; parallelize it. (nvcc host code gets no
    // -fopenmp, so use std::thread rather than OpenMP.)
    //
    // 48-byte POD key hashed/compared by its exact fp32 bit pattern -- same
    // dedup semantics as the old std::string key, but no per-cell heap alloc.
    struct Key { FDTD_FLOAT v[12]; };
    struct KHash {
        size_t operator()(const Key& k) const {
            const unsigned char* p = (const unsigned char*)k.v;
            size_t h = 1469598103934665603ULL;        // FNV-1a over the 48 bytes
            for (size_t i = 0; i < sizeof(k.v); ++i) { h ^= p[i]; h *= 1099511628211ULL; }
            return h;
        }
    };
    struct KEq {
        bool operator()(const Key& a, const Key& b) const {
            return memcmp(a.v, b.v, sizeof(a.v)) == 0;   // bitwise, matches old key
        }
    };
    auto loadKey = [&](int c) {
        Key k;
        k.v[0]=vv[c*3+0];  k.v[1]=vi[c*3+0];
        k.v[2]=vv[c*3+1];  k.v[3]=vi[c*3+1];
        k.v[4]=vv[c*3+2];  k.v[5]=vi[c*3+2];
        k.v[6]=ii[c*3+0];  k.v[7]=iv[c*3+0];
        k.v[8]=ii[c*3+1];  k.v[9]=iv[c*3+1];
        k.v[10]=ii[c*3+2]; k.v[11]=iv[c*3+2];
        return k;
    };

    double _cc_t0 = 0; bool _cc_prof = getenv("OPENEMS_PROF");
    if (_cc_prof) { timeval _t; gettimeofday(&_t,NULL); _cc_t0 = _t.tv_sec + 1e-6*_t.tv_usec; }

    std::vector<unsigned int> index(num_cells);
    std::vector<FDTD_FLOAT> tbl_vvvi;
    std::vector<FDTD_FLOAT> tbl_iiiv;

    unsigned int hw = std::thread::hardware_concurrency();
    int nthreads = (hw == 0) ? 1 : (int)hw;
    if (nthreads > 8) nthreads = 8;              // cap: the serial merge grows with thread count
    if (num_cells < 200000) nthreads = 1;        // small mesh: thread startup not worth it
    if (const char* e = getenv("OPENEMS_COEFF_THREADS")) {   // override (A/B, debug)
        int v = atoi(e);
        if (v >= 1) nthreads = v;
    }

    std::vector<int> rstart(nthreads + 1);
    for (int t = 0; t <= nthreads; ++t)
        rstart[t] = (int)((long long)num_cells * t / nthreads);

    // Pass 1 (parallel): each thread dedups its own cell range into a local table.
    // index[c] temporarily holds the thread-LOCAL unique index.
    std::vector<std::vector<Key>> loc_uniq(nthreads);
    {
        std::vector<std::thread> pool;
        for (int t = 0; t < nthreads; ++t)
            pool.emplace_back([&, t]() {
                std::unordered_map<Key, unsigned int, KHash, KEq> lmap;
                std::vector<Key>& lu = loc_uniq[t];
                for (int c = rstart[t]; c < rstart[t+1]; ++c) {
                    Key k = loadKey(c);
                    auto it = lmap.find(k);
                    unsigned int li;
                    if (it == lmap.end()) { li = (unsigned int)lu.size(); lu.push_back(k); lmap.emplace(k, li); }
                    else li = it->second;
                    index[c] = li;
                }
            });
        for (auto& th : pool) th.join();
    }

    // Merge (serial, but only over the few thread-local uniques): fold the local
    // tables into one globally-deduped table so the on-device table stays as
    // small as the serial build produced -- run-time is unaffected. remap[t][li]
    // maps thread t's local index to the global index.
    std::unordered_map<Key, unsigned int, KHash, KEq> gmap;
    std::vector<std::vector<unsigned int>> remap(nthreads);
    for (int t = 0; t < nthreads; ++t) {
        remap[t].resize(loc_uniq[t].size());
        for (size_t li = 0; li < loc_uniq[t].size(); ++li) {
            const Key& k = loc_uniq[t][li];
            auto it = gmap.find(k);
            unsigned int g;
            if (it == gmap.end()) {
                g = (unsigned int)(tbl_vvvi.size() / 6);
                for (int j = 0; j < 6; ++j) tbl_vvvi.push_back(k.v[j]);
                for (int j = 0; j < 6; ++j) tbl_iiiv.push_back(k.v[6 + j]);
                gmap.emplace(k, g);
            } else g = it->second;
            remap[t][li] = g;
        }
    }

    // Pass 2 (parallel): rewrite each cell's local index to its global index.
    {
        std::vector<std::thread> pool;
        for (int t = 0; t < nthreads; ++t)
            pool.emplace_back([&, t]() {
                const std::vector<unsigned int>& rm = remap[t];
                for (int c = rstart[t]; c < rstart[t+1]; ++c) index[c] = rm[index[c]];
            });
        for (auto& th : pool) th.join();
    }

    unsigned int uniq = (unsigned int)(tbl_vvvi.size() / 6);
    *num_unique = uniq;

    // If the unique count fits in 16 bits (it does for every realistic mesh --
    // tens of thousands of unique coeffs), store the per-cell index as uint16 to
    // halve the index traffic that both update kernels read every timestep.
    // Fall back to uint32 for pathological heterogeneity.
    *index_u16 = (uniq <= 65535u);
    printf("%scoeff compression: %u unique / %d cells (%.2fx dedup, table %.2f MB, %s index)\n",
           tag, uniq, num_cells, (double)num_cells / uniq,
           uniq * 12.0 * sizeof(FDTD_FLOAT) / 1e6, *index_u16 ? "u16" : "u32");
    if (_cc_prof) { timeval _t; gettimeofday(&_t,NULL);
        fprintf(stderr, "[PROF] engine %scoeff compression (%d threads): %.2fs\n", tag,
                nthreads, (_t.tv_sec + 1e-6*_t.tv_usec) - _cc_t0); }
    fflush(stdout);

    if (*index_u16)
    {
        std::vector<unsigned short> idx16(num_cells);
        for (int c = 0; c < num_cells; ++c) idx16[c] = (unsigned short)index[c];
        checkCuda(cudaMalloc(d_index, (size_t)num_cells * sizeof(unsigned short)));
        checkCuda(cudaMemcpy(*d_index, idx16.data(), (size_t)num_cells * sizeof(unsigned short), cudaMemcpyHostToDevice));
    }
    else
    {
        checkCuda(cudaMalloc(d_index, (size_t)num_cells * sizeof(unsigned int)));
        checkCuda(cudaMemcpy(*d_index, index.data(), (size_t)num_cells * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc(d_vv_vi, (size_t)uniq * 6 * sizeof(FDTD_FLOAT)));
    checkCuda(cudaMalloc(d_ii_iv, (size_t)uniq * 6 * sizeof(FDTD_FLOAT)));
    checkCuda(cudaMemcpy(*d_vv_vi, tbl_vvvi.data(), (size_t)uniq * 6 * sizeof(FDTD_FLOAT), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(*d_ii_iv, tbl_iiiv.data(), (size_t)uniq * 6 * sizeof(FDTD_FLOAT), cudaMemcpyHostToDevice));
}

#endif // ENGINE_CUDA_COEFF_H
