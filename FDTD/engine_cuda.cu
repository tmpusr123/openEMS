
/*
 * Copyright (C) 2010 Thorsten Liebig (Thorsten.Liebig@gmx.de)
 * [License text omitted for brevity, remains unchanged]
 */

#include "engine_cuda.h"
#include "extensions/engine_extension.h"
#include "extensions/operator_extension.h"
#include "tools/array_ops.h"

#include <cooperative_groups.h>

#include <cuda_runtime.h>
#include <sys/time.h>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstring>
#include <cstdlib>
#include <thread>

#include "hemi/grid_stride_range.h"
#include "hemi/launch.h"

#include "tools/cuda/check.h"

using namespace std;


#define USE_UNIFIED_MEM

#define flat_index(X, Y, Z, D)     ((X) * D[1] * D[2] + ((Y) * D[2]) + (Z))
#define THREADS     1024
// Update kernels use no shared memory and 40 regs/thread; 1024-thread blocks
// cap occupancy at 1 block/SM (66%). Smaller blocks -> 100% theoretical occupancy.
#define UPDATE_THREADS  256



// Kernel for voltage updates with flat arrays
template<typename IdxT>
__global__
void updateVoltagesKernel(FDTD_FLOAT * __restrict__ volt, const FDTD_FLOAT * __restrict__ curr, const IdxT * __restrict__ op_index, const FDTD_FLOAT * __restrict__ vv_vi_table, int N, dim3 dim)
{
    for (auto cell : hemi::grid_stride_range(0, N)) {

        int offs = cell * 3;
        int x = cell / (dim.y * dim.z);
        int y = (cell / dim.z) % dim.y;
        int z = cell % dim.z;

        const FDTD_FLOAT* ix = curr + ((x != 0) ? offs - dim.y * dim.z * 3 : offs);
        const FDTD_FLOAT* iy = curr + ((y != 0) ? offs - dim.z * 3 : offs);
        const FDTD_FLOAT* iz = curr + ((z != 0) ? offs - 3 : offs);

        // Compressed coefficients: one 4-byte index per cell into a small
        // unique-coefficient table (L2-resident), instead of 24 B of vv/vi.
        float2 vvvi[3];
        const FDTD_FLOAT* c = vv_vi_table + (size_t)op_index[cell] * 6;

        vvvi[0] = *(float2*)(c);
        vvvi[1] = *(float2*)(c + 2);
        vvvi[2] = *(float2*)(c + 4);

        FDTD_FLOAT i[3];
        const FDTD_FLOAT *p = curr + offs;
        i[0]= *p++;
        i[1] = *p++;
        i[2] = *p++;

        // nbr cells in x, y, z direction
        vvvi[0].y *= (i[2] - iy[2] - i[1] + iz[1]);
        vvvi[1].y *= (i[0] - iz[0] - i[2] + ix[2]);
        vvvi[2].y *= (i[1] -ix[1] - i[0] + iy[0]); 

        // update x
        FDTD_FLOAT* v= volt + offs;
        v[0]  = v[0] * vvvi[0].x + vvvi[0].y; 
        // update y
        v[1] = v[1] * vvvi[1].x + vvvi[1].y;
        // update z
        v[2] = v[2] * vvvi[2].x +  vvvi[2].y;
    }
}


// Kernel for current updates
template<typename IdxT>
__global__
void updateCurrentsKernel(FDTD_FLOAT * __restrict__ curr, const FDTD_FLOAT * __restrict__ volt, const IdxT * __restrict__ op_index, const FDTD_FLOAT * __restrict__ ii_iv_table, int N, dim3 dim)
{
    for (auto cell : hemi::grid_stride_range(0, N)) {

        int offs = cell * 3;
        int y = (cell / dim.z) % dim.y;
        int z = cell % dim.z;

        if ((y < dim.y - 1) && (z < dim.z - 1)) {

            volt += offs;
            // next nbr cells in x, y, z direction
            const FDTD_FLOAT* vx = volt + (dim.y * dim.z * 3);
            const FDTD_FLOAT* vy = volt + (3 * dim.z);
            const FDTD_FLOAT* vz = volt + 3;

            // Compressed coefficients (same per-cell index as the voltage kernel).
            float2 iiiv[3];
            const FDTD_FLOAT* c = ii_iv_table + (size_t)op_index[cell] * 6;

            iiiv[0] = *(float2 *)(&c[0]);
            iiiv[1] = *(float2 *)(&c[2]);
            iiiv[2] = *(float2 *)(&c[4]);

            FDTD_FLOAT v[3];
            v[0] = volt[0];
            v[1] = volt[1];
            v[2] = volt[2];

            iiiv[0].y *= (v[2] - vy[2] - v[1] + vz[1]);
            iiiv[1].y *= (v[0] - vz[0] - v[2] + vx[2]);
            iiiv[2].y *= (v[1] - vx[1] - v[0] + vy[0]);

            FDTD_FLOAT* i = curr + offs;
            // update x
            i[0] = i[0] * iiiv[0].x + iiiv[0].y;
            // update y
            i[1] = i[1] * iiiv[1].x + iiiv[1].y;
            // update z
            i[2] = i[2] * iiiv[2].x + iiiv[2].y;
        }
    }
}

__global__
void addInKernel(FDTD_FLOAT *p, int cell, int n, FDTD_FLOAT val)
{
    p[cell * 3 + n] += val;
}

// Advance the device-side timestep counter by one. Runs at the tail of every
// captured timestep so the excitation kernels see the correct numTS on replay.
__global__
void incrementNumTSKernel(int *d_numTS)
{
    *d_numTS += 1;
}

// Seed the device counter at the start of a chunk (launched outside the graph).
__global__
void setNumTSKernel(int *d_numTS, int val)
{
    *d_numTS = val;
}

__device__ void wrapReduce(volatile double *vv, volatile double *ii, int tid)
{
    vv[tid] += vv[(tid + 32)];
    ii[tid] += ii[(tid + 32)];
    vv[tid] += vv[(tid + 16)];
    ii[tid] += ii[(tid + 16)];
    vv[tid] += vv[(tid + 8)];
    ii[tid] += ii[(tid + 8)];
    vv[tid] += vv[(tid + 4)];
    ii[tid] += ii[(tid + 4)];
    vv[tid] += vv[(tid + 2)];
    ii[tid] += ii[(tid + 2)];
    vv[tid] += vv[(tid + 1)];
    ii[tid] += ii[(tid + 1)];
}

// Multi-block grid reduction: every block grid-strides over the cells, reduces
// its partial E/H energy in shared memory, and atomically accumulates the
// weighted result into *p_sum (zeroed by the host before launch). Replaces the
// old single-block version that left 45 of 46 SMs idle (35 ms -> ~1.5 ms at
// 26M cells). The energy is a stop-criterion diagnostic, not physics; the
// block-level accumulation order differs from the old kernel only in fp
// rounding of that diagnostic.
__global__ void calcFastEnergyKernel(FDTD_FLOAT *volt, FDTD_FLOAT *curr, double *p_sum, dim3 dim)
{
    __shared__ double vv[THREADS]; // shared memory for energy calculation, between threads in a block
    __shared__ double ii[THREADS]; // shared memory for energy calculation, between threads in a block

    double local_vv = 0.0;
    double local_ii = 0.0;
    int tid = threadIdx.x;

    // Process all cells in grid strides
    int total = dim.x * dim.y * dim.z;
    for (int cell = blockIdx.x * THREADS + tid; cell < total; cell += gridDim.x * THREADS) {
        int x = cell / (dim.y * dim.z);
        int y = (cell / dim.z) % dim.y;
        int z = cell % dim.z;
        if (x < dim.x - 1 && y < dim.y - 1 && z < dim.z - 1) {
            int offs = cell * 3;
            FDTD_FLOAT *v = volt + offs;
            FDTD_FLOAT *i = curr + offs;

            local_vv += v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            local_ii += i[0] * i[0] + i[1] * i[1] + i[2] * i[2];
        }
    }
    vv[tid] = local_vv;
    ii[tid] = local_ii;
    __syncthreads();

    // Reduction
    for (int s = THREADS / 2; s > 32; s >>= 1) {
        if (tid < s) {
            vv[tid] += vv[tid + s];
            ii[tid] += ii[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32) {
        wrapReduce(vv, ii, tid);
    }
    __syncthreads();

    if (tid == 0) {
        atomicAdd(p_sum, vv[0] * __EPS0__ + ii[0] * __MUE0__);
    }
}


Engine_cuda* Engine_cuda::New(const Operator_CUDA* op, unsigned int cuda_device_number) {
    cout << "Create CUDA FDTD engine" << endl;
    Engine_cuda* e = new Engine_cuda(op);
    e->setCUDAdevice(cuda_device_number);
    e->Init();
    return e;
}

Engine_cuda::Engine_cuda(const Operator_CUDA* op)  : Engine(op)
{
    m_type = CUDA;
    numTS = 0;
    Op = op;

    // Initialize host/device sync state (were uninitialized -> UB on first IterateTS)
    m_volt_updated = 0;
    m_curr_updated = 0;
    m_volt_updated_by_host = 0;
    m_curr_updated_by_host = 0;
    m_host_data_locked = false;

    d_numTS = NULL;
    m_graph = NULL;
    m_graphExec = NULL;
    m_graph_ready = false;

    m_rb_fullmode = false;

    d_op_index = NULL;
    d_vv_vi_table = NULL;
    d_ii_iv_table = NULL;
    m_num_unique_coeff = 0;
    m_index_u16 = false;

    // Null everything Reset() touches: the mgpu subclass never runs this
    // class's Init(), so its dtor chain must find clean members here.
    volt_array = NULL;
    curr_array = NULL;
    m_energy_sum = NULL;
    d_op_index = NULL;
    d_fastEnergy = NULL;

    cout << "Engine CUDA construct type=" << m_type << endl;
}

void Engine_cuda::setCUDAdevice(unsigned int cuda_device_number)	   
{
    m_cuda_device_number = cuda_device_number;
}


Engine_cuda::~Engine_cuda()
{
    this->Reset();
}



// Deduplicate each cell's 12-float coefficient set (vv,vi interleaved for the
// voltage kernel; ii,iv for the current kernel, in exactly the order the kernels
// read them) into two small unique-coefficient tables plus a per-cell index, and
// upload all three to the device. Homogeneous regions (free space, uniform mesh)
// collapse to a handful of unique entries, so the tables stay L2-resident and the
// update kernels move a 4-byte index instead of 24 bytes of coefficients per cell.
static void build_compressed_coeff(
    const FDTD_FLOAT *vv, const FDTD_FLOAT *vi,
    const FDTD_FLOAT *ii, const FDTD_FLOAT *iv,
    int num_cells,
    void **d_index, bool *index_u16, FDTD_FLOAT **d_vv_vi, FDTD_FLOAT **d_ii_iv,
    unsigned int *num_unique)
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
    printf("coeff compression: %u unique / %d cells (%.2fx dedup, table %.2f MB, %s index)\n",
           uniq, num_cells, (double)num_cells / uniq,
           uniq * 12.0 * sizeof(FDTD_FLOAT) / 1e6, *index_u16 ? "u16" : "u32");
    if (_cc_prof) { timeval _t; gettimeofday(&_t,NULL);
        fprintf(stderr, "[PROF] engine coeff compression (%d threads): %.2fs\n",
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

void Engine_cuda::Init() {
	   

    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if (nDevices <= 0) {
        throw std::runtime_error("NO CUDA device found");
    }
    if (m_cuda_device_number >= nDevices) {
        cout << "cuda device out of range " << m_cuda_device_number << " / " << nDevices << endl;
        throw std::runtime_error("CUDA device number out of range");
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, m_cuda_device_number);
    cout << "  Running on device: " << prop.name << endl;
    cout << "    max block dimensions " \
        << prop.maxThreadsDim[0] << "," \
        << prop.maxThreadsDim[1] << "," \
        << prop.maxThreadsDim[2] << endl;

    cout << "    max block dimensions " \
        << prop.maxGridSize[0] << "," \
        << prop.maxGridSize[1] << "," \
        << prop.maxGridSize[2] << endl;
   
    cudaSetDevice(m_cuda_device_number);
    cudaDeviceGetAttribute(&m_supports_coop_launch, cudaDevAttrCooperativeLaunch, m_cuda_device_number);

    m_dim = dim3(numLines[0], numLines[1], numLines[2]);

	numTS = 0;
    int num_cells = numLines[0] * numLines[1] * numLines[2];

    printf("simulation dim: %d, %d, %d\n", m_dim.x, m_dim.y, m_dim.z); fflush(stdout);

    // Allocate GPU memory
	volt_array = new CudaHelper::Array<FDTD_FLOAT>(num_cells * 3);
	curr_array = new CudaHelper::Array<FDTD_FLOAT>(num_cells * 3);
    m_energy_sum =  new CudaHelper::Array<double>(1);

    volt_array->clear();
    curr_array->clear();

    checkCuda(cudaMalloc(&d_numTS, sizeof(int)));

    build_compressed_coeff(Op->vv_ptr->data(), Op->vi_ptr->data(),
                           Op->ii_ptr->data(), Op->iv_ptr->data(),
                           num_cells,
                           &d_op_index, &m_index_u16, &d_vv_vi_table, &d_ii_iv_table,
                           &m_num_unique_coeff);

    InitExtensions();
    SortExtensionByPriority();

    // Safety net for the whole "un-ported extension silently no-ops on GPU" bug
    // class (the exact failure that hid the Mur / lumped-RLC / dispersive
    // breakage before those were ported): any engine extension lacking a real
    // on-device implementation would run its host-side hooks against the
    // device-resident fields that RunOneTimestep deliberately never syncs, so
    // its physics is SILENTLY dropped with no error. Warn loudly at setup.
    for (size_t n = 0; n < m_Eng_exts.size(); ++n)
    {
        if (!m_Eng_exts.at(n)->IsCUDACapable())
        {
            std::cerr << "openEMS CUDA engine WARNING: extension '"
                      << m_Eng_exts.at(n)->GetExtensionName()
                      << "' has no CUDA implementation -- it runs against un-synced "
                         "device fields and its effect is SILENTLY IGNORED on the GPU. "
                         "Results may be wrong; use the CPU engine for this model."
                      << std::endl;
        }
    }
}

void Engine_cuda::Reset() {
    FreeFieldGathers();
    if (m_graph_ready) {
        cudaGraphExecDestroy(m_graphExec);
        cudaGraphDestroy(m_graph);
        m_graph_ready = false;
    }
    if (d_numTS)        cudaFree(d_numTS);
    d_numTS = NULL;
    if (d_rb_list)      cudaFree(d_rb_list);
    if (d_rb_stage)     cudaFree(d_rb_stage);
    if (h_rb_stage)     cudaFreeHost(h_rb_stage);
    d_rb_list = NULL; d_rb_stage = NULL; h_rb_stage = NULL;
    m_rb_capacity = 0; m_rb_count = 0; m_rb_dirty = true; m_rb_pending = false;
    if (d_op_index)     cudaFree(d_op_index);
    if (d_vv_vi_table)  cudaFree(d_vv_vi_table);
    if (d_ii_iv_table)  cudaFree(d_ii_iv_table);
    d_op_index = NULL; d_vv_vi_table = NULL; d_ii_iv_table = NULL;

    if (volt_array)     delete volt_array;
    if (curr_array)     delete curr_array;
    if (m_energy_sum)   delete m_energy_sum;

    ClearExtensions();
}


void Engine_cuda::UpdateVoltages(unsigned int startX, unsigned int numX) {
    // Copy current data to GPU
    int N = numX * numLines[1] * numLines[2];

    // Launch kernel
    int blocks = (N  + UPDATE_THREADS - 1) / UPDATE_THREADS;
    dim3 dim(numLines[0], numLines[1], numLines[2]);
    if (m_index_u16)
        updateVoltagesKernel<<< blocks, UPDATE_THREADS >>>(volt_array->device_data(), (const FDTD_FLOAT*)curr_array->device_data(),
                (const unsigned short*)d_op_index, d_vv_vi_table, N, dim);
    else
        updateVoltagesKernel<<< blocks, UPDATE_THREADS >>>(volt_array->device_data(), (const FDTD_FLOAT*)curr_array->device_data(),
                (const unsigned int*)d_op_index, d_vv_vi_table, N, dim);

    //checkCudaErrors();
}

void Engine_cuda::UpdateCurrents(unsigned int startX, unsigned int numX) {
    // Copy voltage data to GPU

    int N = numX * numLines[1] * numLines[2];

    // Launch kernel
    int blocks = (N  + UPDATE_THREADS - 1) / UPDATE_THREADS;
    dim3 dim(numLines[0], numLines[1], numLines[2]);

    if (m_index_u16)
        updateCurrentsKernel<<< blocks, UPDATE_THREADS >>>(curr_array->device_data(), (const FDTD_FLOAT*)volt_array->device_data(),
                (const unsigned short*)d_op_index, d_ii_iv_table, N, dim);
    else
        updateCurrentsKernel<<< blocks, UPDATE_THREADS >>>(curr_array->device_data(), (const FDTD_FLOAT*)volt_array->device_data(),
                (const unsigned int*)d_op_index, d_ii_iv_table, N, dim);
    //checkCudaErrors();
}

void Engine_cuda::AddVolt(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value)
{
    int cell = flat_index(pos[0], pos[1], pos[2], numLines);
    addInKernel<<< 1, 1>>>(volt_array->device_data(), cell, n, value);

    //checkCudaErrors();
}

void Engine_cuda::AddCurr(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value)
{
    int cell = flat_index(pos[0], pos[1], pos[2], numLines);
    addInKernel<<<1, 1>>>(curr_array->device_data(), cell, n, value);

    //checkCudaErrors();
}


// Whole-grid energy reduction straight off the device (syncs via load_to_host).
double Engine_cuda::ComputeDeviceEnergy()
{
    // Runs outside the captured graph, so a memset + multi-block launch is fine.
    int total = numLines[0] * numLines[1] * numLines[2];
    int blocks = (total + THREADS - 1) / THREADS;
    if (blocks > 184) blocks = 184;   // ~4 blocks/SM saturates the reduction
    checkCuda(cudaMemsetAsync(m_energy_sum->device_data(), 0, sizeof(double), cudaStreamPerThread));
    calcFastEnergyKernel<<<blocks, THREADS>>>(volt_array->device_data(), curr_array->device_data(), m_energy_sum->device_data(), m_dim);

    CudaHelper::check_cuda();
    //checkCuda(cudaDeviceSynchronize());

    m_energy_sum->load_to_host();
    return m_energy_sum->host_data()[0];
}

double Engine_cuda::CalcFastEnergy()
{
    // With GPU field gathers active the loop pipelines and the device may be
    // computing the NEXT chunk when the run loop asks for energy. Return the
    // value cached at this chunk's finish (device-idle, chunk-consistent) so
    // the reading matches the fields the host is processing -- and so it does
    // not stall the pipeline by syncing on the in-flight chunk.
    if (m_energy_valid) return m_energy_cache;
    return ComputeDeviceEnergy();
}

// Other methods (InitExtensions, SortExtensionByPriority, etc.) remain unchanged unless extensions need CUDA support

// Fetch a single cell's volt+curr into the host mirror the first time a probe
// asks for it. Subsequent chunks refresh it via SelectiveReadback, so this
// on-demand copy only happens once per distinct probed cell. If the probed
// footprint grows past ~5% of the grid (e.g. a large field dump), give up and
// switch to whole-array readback.
void Engine_cuda::EnsureCellHost(int cell) const
{
    if (m_rb_fullmode) return;
    // dump extraction calls GetVolt/GetCurr from parallel threads; guard the
    // cell-learning state (fullmode fast path above stays lock-free)
    std::lock_guard<std::mutex> lk(m_rb_mtx);
    if (m_rb_cells.find(cell) != m_rb_cells.end()) return;   // already refreshed

    m_rb_cells.insert(cell);
    m_rb_dirty = true;   // batched readback must re-upload its cell list

    int o = cell * 3;
    cudaMemcpy(volt_array->host_data() + o, volt_array->device_data() + o,
               3 * sizeof(FDTD_FLOAT), cudaMemcpyDeviceToHost);
    cudaMemcpy(curr_array->host_data() + o, curr_array->device_data() + o,
               3 * sizeof(FDTD_FLOAT), cudaMemcpyDeviceToHost);

    size_t num_cells = (size_t)numLines[0] * numLines[1] * numLines[2];
    if (m_rb_cells.size() > num_cells / 20)
        m_rb_fullmode = true;
}

// Gather the tracked cells' volt+curr into a contiguous staging buffer
// (6 floats per cell: volt xyz then curr xyz).
__global__ void gatherReadbackKernel(const FDTD_FLOAT *volt, const FDTD_FLOAT *curr,
                                     const int *cells, FDTD_FLOAT *stage, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int o = cells[i] * 3;
    FDTD_FLOAT *s = stage + i * 6;
    s[0] = volt[o]; s[1] = volt[o + 1]; s[2] = volt[o + 2];
    s[3] = curr[o]; s[4] = curr[o + 1]; s[5] = curr[o + 2];
}

// Refresh every tracked cell's volt+curr in the host mirror: one gather kernel
// + ONE D2H copy into pinned staging (instead of 2 tiny cudaMemcpyAsync per
// cell, whose per-copy launch latency dominated port-driven chunks). The
// chunk-end cudaDeviceSynchronize retires the copy; SelectiveReadbackFinish
// then scatters into the host mirror.
void Engine_cuda::SelectiveReadback()
{
    int n = (int)m_rb_cells.size();
    if (n == 0) return;

    if (m_rb_dirty) {
        m_rb_list.assign(m_rb_cells.begin(), m_rb_cells.end());
        if ((size_t)n > m_rb_capacity) {
            if (d_rb_list)  cudaFree(d_rb_list);
            if (d_rb_stage) cudaFree(d_rb_stage);
            if (h_rb_stage) cudaFreeHost(h_rb_stage);
            m_rb_capacity = (size_t)n * 2;   // headroom for a growing footprint
            checkCuda(cudaMalloc(&d_rb_list,  m_rb_capacity * sizeof(int)));
            checkCuda(cudaMalloc(&d_rb_stage, m_rb_capacity * 6 * sizeof(FDTD_FLOAT)));
            checkCuda(cudaHostAlloc(&h_rb_stage, m_rb_capacity * 6 * sizeof(FDTD_FLOAT), cudaHostAllocDefault));
        }
        checkCuda(cudaMemcpyAsync(d_rb_list, m_rb_list.data(), n * sizeof(int),
                                  cudaMemcpyHostToDevice, cudaStreamPerThread));
        m_rb_dirty = false;
    }
    m_rb_count = n;

    int blocks = (n + 127) / 128;
    gatherReadbackKernel<<<blocks, 128>>>(volt_array->device_data(), curr_array->device_data(),
                                          d_rb_list, d_rb_stage, n);
    cudaMemcpyAsync(h_rb_stage, d_rb_stage, (size_t)n * 6 * sizeof(FDTD_FLOAT),
                    cudaMemcpyDeviceToHost, cudaStreamPerThread);
    m_rb_pending = true;
}

// Scatter the staged values into the host field mirror. Must run after the
// chunk-end cudaDeviceSynchronize has retired the async D2H copy.
void Engine_cuda::SelectiveReadbackFinish()
{
    if (!m_rb_pending) return;
    m_rb_pending = false;
    FDTD_FLOAT *vh = volt_array->host_data();
    FDTD_FLOAT *ch = curr_array->host_data();
    for (int i = 0; i < m_rb_count; ++i) {
        int o = m_rb_list[i] * 3;
        const FDTD_FLOAT *s = h_rb_stage + (size_t)i * 6;
        vh[o] = s[0]; vh[o + 1] = s[1]; vh[o + 2] = s[2];
        ch[o] = s[3]; ch[o + 1] = s[4]; ch[o + 2] = s[5];
    }
}

// One timestep of updates: PML pre/post + core Yee + excitation for both
// fields, then advance the device timestep counter. Deliberately free of any
// host<->device copies or syncs so the whole sequence can be stream-captured
// into a CUDA graph and replayed once per timestep (collapsing the ~7 kernel
// launches + C++ extension dispatch into a single graph launch).
void Engine_cuda::RunOneTimestep()
{
    DoPreVoltageUpdates();
    UpdateVoltages(0, numLines[0]);
    DoPostVoltageUpdates();
    Apply2Voltages();

    DoPreCurrentUpdates();
    UpdateCurrents(0, numLines[0] - 1);
    DoPostCurrentUpdates();
    Apply2Current();

    incrementNumTSKernel<<<1, 1>>>(d_numTS);
}

bool Engine_cuda::IterateTS(unsigned int iterTS) {
    LaunchChunkBody(iterTS);
    FinishChunkBody(iterTS);
    return true;
}

// Pipelined chunks (see engine_cuda.h): in full-mirror mode queue the chunk
// and return immediately -- the caller processes the previous chunk's host
// state while the GPU computes. In selective-readback mode defer to the base
// class (synchronous execution inside WaitChunk), since probes may still
// learn new mirror cells, which requires an idle device at the presented
// timestep.
void Engine_cuda::LaunchChunkAsync(unsigned int iterTS) {
    // Pipeline when there is host-side work to overlap with the next chunk:
    // either a full-mirror readback, or GPU field gathers (whose HDF5 writes
    // run on the host while the next chunk computes).
    if (!m_rb_fullmode && m_gathers.empty()) {
        Engine::LaunchChunkAsync(iterTS);
        return;
    }
    LaunchChunkBody(iterTS);
    // The host mirror still presents the *previous* chunk's state and stays
    // untouched until FinishChunkBody -- reading it during the flight is the
    // whole point of the pipeline, so don't leave it locked.
    m_host_data_locked = false;
    // The pre-chunk host edits are now uploaded; clear the flags so WaitChunk's
    // guard can detect any *new* host write that sneaks in during the flight.
    m_volt_updated = 0;
    m_curr_updated = 0;
    m_async_pending += iterTS;
}

bool Engine_cuda::WaitChunk() {
    if (m_async_pending) {
        // Host field writes while a chunk is in flight would race the queued
        // GPU work; nothing in openEMS's run loop does this -- fail loudly if
        // that ever changes rather than corrupt the fields silently.
        if (m_volt_updated || m_curr_updated) {
            fprintf(stderr, "Engine_cuda::WaitChunk: host wrote fields while an async chunk was in flight -- unsupported, aborting\n");
            abort();
        }
        FinishChunkBody(m_async_pending);
        m_async_pending = 0;
        return true;
    }
    return Engine::WaitChunk();   // run a deferred (selective-mode) chunk
}

// Queue one chunk on the per-thread stream: upload a host-modified mirror,
// seed the device timestep counter, replay the per-timestep graph iterTS
// times. Everything is async -- no host sync, numTS untouched.
void Engine_cuda::LaunchChunkBody(unsigned int iterTS) {
    m_volt_updated_by_host = 1;
    m_curr_updated_by_host = 1;

    if (m_volt_updated) {
        volt_array->load_to_device_async();
    }
    if (m_curr_updated) {
        curr_array->load_to_device_async();
    }
    m_host_data_locked = true;

    // Capture one timestep into a CUDA graph on first use. Grid dims and launch
    // configs are fixed for the whole run, so the graph is reused for every
    // chunk. Capture records the kernel launches without executing them; all
    // iterTS timesteps below are executed via graph replay.
    // (A 2-timestep "graph rotation" fusing PostCurr(t)+PreVolt(t+1) was
    // implemented, validated bit-exact, and measured at 0% -- the graph
    // executor already pipelines these seams -- so it was removed again.)
    if (!m_graph_ready) {
        cudaStreamSynchronize(cudaStreamPerThread);
        checkCuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        RunOneTimestep();
        checkCuda(cudaStreamEndCapture(cudaStreamPerThread, &m_graph));
        checkCuda(cudaGraphInstantiate(&m_graphExec, m_graph, NULL, NULL, 0));
        m_graph_ready = true;
    }

    // Seed the device counter with this chunk's starting timestep, then replay
    // one graph launch per timestep. Each launch advances the counter on-device
    // so the excitation kernels see the correct numTS without host intervention.
    setNumTSKernel<<<1, 1>>>(d_numTS, (int)numTS);

    for (unsigned int iter = 0; iter < iterTS; ++iter) {
        checkCuda(cudaGraphLaunch(m_graphExec, cudaStreamPerThread));
    }
}

// Finish a queued chunk: read the fields back into the host mirror, sync,
// and make the chunk visible (numTS advance, mirror unlocked).
void Engine_cuda::FinishChunkBody(unsigned int iterTS) {
    numTS += iterTS;

    timeval _pt0, _pt1;
    gettimeofday(&_pt0, NULL);

    // Refresh the host field mirror for host-side probes/dumps. Only the cells
    // probes actually read are copied back (learned on demand); a whole-array
    // copy is a huge waste when a lumped port reads a few dozen cells out of
    // millions. Falls back to a full copy once the footprint gets large.
    // Volt is not modified after its last update, so a post-loop refresh matches
    // the old interleaved copy exactly.
    if (m_rb_fullmode) {
        volt_array->load_to_host_async();
        curr_array->load_to_host_async();
    } else {
        SelectiveReadback();
    }

    // On-device field-dump interpolation: the device holds this chunk's fields
    // right now, so gather the registered dump stencils straight from the
    // device arrays (no full-array readback needed). Enqueued on the same
    // stream, completed by the sync below.
    RunFieldGathers();

    // With GPU gathers, dumps no longer force a whole-mirror readback, so the
    // energy end-criterion can't read the mirror. Enqueue a device-side energy
    // reduction here (device holds this chunk's fields), piggybacked on the
    // single sync below -- no extra sync, no pipeline stall. CalcFastEnergy()
    // then returns this cached, chunk-consistent value.
    if (!m_gathers.empty()) {
        int total = numLines[0] * numLines[1] * numLines[2];
        int blocks = (total + THREADS - 1) / THREADS;
        if (blocks > 184) blocks = 184;
        checkCuda(cudaMemsetAsync(m_energy_sum->device_data(), 0, sizeof(double), cudaStreamPerThread));
        calcFastEnergyKernel<<<blocks, THREADS, 0, cudaStreamPerThread>>>(
            volt_array->device_data(), curr_array->device_data(), m_energy_sum->device_data(), m_dim);
        checkCuda(cudaMemcpyAsync(m_energy_sum->host_data(), m_energy_sum->device_data(),
                                  sizeof(double), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    }

    // Async copies land in pinned host memory; sync before host reads them.
    cudaDeviceSynchronize();
    SelectiveReadbackFinish();   // scatter staged probe cells into the mirror

    if (!m_gathers.empty()) {
        m_energy_cache = m_energy_sum->host_data()[0];
        m_energy_valid = true;
    }

    gettimeofday(&_pt1, NULL);
    m_prof_readback += (_pt1.tv_sec - _pt0.tv_sec) + 1e-6 * (_pt1.tv_usec - _pt0.tv_usec);

    m_host_data_locked = false;
    m_volt_updated = 0;
    m_curr_updated = 0;
}

// ---------------------------------------------------------------------------
// On-device field-dump gather (FieldGatherBackend)
// ---------------------------------------------------------------------------

// out[o] = sum over CSR entries [offsets[o],offsets[o+1]) of coeff*src[idx].
// src is the full device volt or curr array; idx already folds in the component
// (idx = (x*ny*nz+y*nz+z)*3 + comp), coeff folds in all geometry/material.
__global__ void fieldGatherKernel(const FDTD_FLOAT* __restrict__ src,
                                  const unsigned int* __restrict__ offsets,
                                  const unsigned int* __restrict__ idx,
                                  const float* __restrict__ coeff,
                                  float* __restrict__ out, int nOut)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= nOut) return;
    unsigned int a = offsets[o], b = offsets[o + 1];
    float v = 0.0f;
    for (unsigned int e = a; e < b; ++e)
        v += coeff[e] * (float)src[idx[e]];
    out[o] = v;
}

void Engine_cuda::RunFieldGathers()
{
    for (size_t g = 0; g < m_gathers.size(); ++g) {
        GpuGather &G = m_gathers[g];
        const FDTD_FLOAT *src = G.useCurr ? GetDeviceCurrData() : GetDeviceVoltData();
        int nOut = (int)G.nOut;
        int block = 128, grid = (nOut + block - 1) / block;
        fieldGatherKernel<<<grid, block, 0, cudaStreamPerThread>>>(
            src, G.d_offsets, G.d_src, G.d_coeff, G.d_out, nOut);
        checkCuda(cudaMemcpyAsync(G.h_out, G.d_out, G.nOut * sizeof(float),
                                  cudaMemcpyDeviceToHost, cudaStreamPerThread));
        G.ts = (long)numTS;   // fields on device correspond to this timestep
    }
}

int Engine_cuda::RegisterFieldGather(const std::vector<unsigned int>& offsets,
                                     const std::vector<unsigned int>& src,
                                     const std::vector<float>& coeff,
                                     bool useCurr, size_t nOut, float** hostOut)
{
    if (offsets.size() != nOut + 1 || src.size() != coeff.size())
        return -1;

    // Budget check: skip GPU gather if it would take too much of the device's
    // free memory (caller then uses the host interpolation path).
    size_t need = offsets.size() * sizeof(unsigned int)
                + src.size() * (sizeof(unsigned int) + sizeof(float))
                + nOut * sizeof(float);
    size_t freeB = 0, totB = 0;
    if (cudaMemGetInfo(&freeB, &totB) != cudaSuccess) { cudaGetLastError(); return -1; }
    if (need > freeB / 2) {
        fprintf(stderr, "Engine_cuda: field-dump gather needs %.0f MB > half of %.0f MB free -- using host path\n",
                need / 1048576.0, freeB / 1048576.0);
        return -1;
    }

    GpuGather G;
    G.nOut = nOut;
    G.useCurr = useCurr;
    G.ts = -1;
    if (cudaMalloc(&G.d_offsets, offsets.size() * sizeof(unsigned int)) != cudaSuccess) { cudaGetLastError(); return -1; }
    if (cudaMalloc(&G.d_src, src.size() * sizeof(unsigned int)) != cudaSuccess ||
        cudaMalloc(&G.d_coeff, coeff.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&G.d_out, nOut * sizeof(float)) != cudaSuccess ||
        cudaHostAlloc(&G.h_out, nOut * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
        cudaGetLastError();
        // best-effort cleanup of whatever succeeded
        cudaFree(G.d_offsets); cudaFree(G.d_src); cudaFree(G.d_coeff); cudaFree(G.d_out);
        return -1;
    }
    checkCuda(cudaMemcpy(G.d_offsets, offsets.data(), offsets.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(G.d_src, src.data(), src.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(G.d_coeff, coeff.data(), coeff.size() * sizeof(float), cudaMemcpyHostToDevice));
    memset(G.h_out, 0, nOut * sizeof(float));

    *hostOut = G.h_out;
    m_gathers.push_back(G);
    return (int)m_gathers.size() - 1;
}

long Engine_cuda::GetGatherTS(int id) const
{
    if (id < 0 || id >= (int)m_gathers.size()) return -1;
    return m_gathers[id].ts;
}

void Engine_cuda::FreeFieldGathers()
{
    for (size_t g = 0; g < m_gathers.size(); ++g) {
        cudaFree(m_gathers[g].d_offsets);
        cudaFree(m_gathers[g].d_src);
        cudaFree(m_gathers[g].d_coeff);
        cudaFree(m_gathers[g].d_out);
        cudaFreeHost(m_gathers[g].h_out);
    }
    m_gathers.clear();
}

