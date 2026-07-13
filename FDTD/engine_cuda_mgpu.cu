/*
*	Multi-GPU CUDA FDTD engine: contiguous slab decomposition along x with
*	per-timestep, event-choreographed halo exchange. See engine_cuda_mgpu.h for
*	the layout and choreography; the per-cell update math is copied verbatim
*	from engine_cuda.cu so results are bit-identical to the single-GPU engine.
*/

#include "engine_cuda_mgpu.h"
#include "operator_cuda.h"
#include "extensions/engine_extension.h"

#include <cuda_runtime.h>
#include <cstring>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <iostream>

#include "hemi/hemi_error.h"
#include "hemi/grid_stride_range.h"
#include "tools/constants.h"

#define MG_THREADS 256

// defined (non-static) in engine_cuda.cu
__global__ void incrementNumTSKernel(int *d_numTS);
__global__ void setNumTSKernel(int *d_numTS, int val);

// ---------------------------------------------------------------------------
// Kernels (slab-local twins of the single-GPU kernels; identical per-cell math)
// ---------------------------------------------------------------------------

// Voltage update over the slab's real cells. cell indexes the REAL region
// (0..nx_real*ny*nz); local arrays have one ghost plane in front, so the local
// offset is (cell + plane). The global x==0 self-fallback applies only on the
// first slab; every other left-neighbor read may fall into the ghost plane.
template<typename IdxT>
__global__
void updateVoltagesMgKernel(FDTD_FLOAT * __restrict__ volt, const FDTD_FLOAT * __restrict__ curr,
                            const IdxT * __restrict__ op_index, const FDTD_FLOAT * __restrict__ vv_vi_table,
                            int N, int ny, int nz, bool is_first)
{
    int plane = ny * nz;
    for (auto cell : hemi::grid_stride_range(0, N)) {

        int offs = (cell + plane) * 3;
        int xr = cell / plane;
        int y = (cell / nz) % ny;
        int z = cell % nz;

        const FDTD_FLOAT* ix = curr + ((!is_first || xr != 0) ? offs - plane * 3 : offs);
        const FDTD_FLOAT* iy = curr + ((y != 0) ? offs - nz * 3 : offs);
        const FDTD_FLOAT* iz = curr + ((z != 0) ? offs - 3 : offs);

        float2 vvvi[3];
        const FDTD_FLOAT* c = vv_vi_table + (size_t)op_index[cell] * 6;

        vvvi[0] = *(float2*)(c);
        vvvi[1] = *(float2*)(c + 2);
        vvvi[2] = *(float2*)(c + 4);

        FDTD_FLOAT i[3];
        const FDTD_FLOAT *p = curr + offs;
        i[0] = *p++;
        i[1] = *p++;
        i[2] = *p++;

        vvvi[0].y *= (i[2] - iy[2] - i[1] + iz[1]);
        vvvi[1].y *= (i[0] - iz[0] - i[2] + ix[2]);
        vvvi[2].y *= (i[1] - ix[1] - i[0] + iy[0]);

        FDTD_FLOAT* v = volt + offs;
        v[0] = v[0] * vvvi[0].x + vvvi[0].y;
        v[1] = v[1] * vvvi[1].x + vvvi[1].y;
        v[2] = v[2] * vvvi[2].x + vvvi[2].y;
    }
}

// Current update; N covers nx_real - (is_last ? 1 : 0) planes (the single-GPU
// engine iterates numLines[0]-1 planes, i.e. skips the global last plane).
template<typename IdxT>
__global__
void updateCurrentsMgKernel(FDTD_FLOAT * __restrict__ curr, const FDTD_FLOAT * __restrict__ volt,
                            const IdxT * __restrict__ op_index, const FDTD_FLOAT * __restrict__ ii_iv_table,
                            int N, int ny, int nz)
{
    int plane = ny * nz;
    for (auto cell : hemi::grid_stride_range(0, N)) {

        int offs = (cell + plane) * 3;
        int y = (cell / nz) % ny;
        int z = cell % nz;

        if ((y < ny - 1) && (z < nz - 1)) {

            const FDTD_FLOAT* vbase = volt + offs;
            const FDTD_FLOAT* vx = vbase + (plane * 3);
            const FDTD_FLOAT* vy = vbase + (3 * nz);
            const FDTD_FLOAT* vz = vbase + 3;

            float2 iiiv[3];
            const FDTD_FLOAT* c = ii_iv_table + (size_t)op_index[cell] * 6;

            iiiv[0] = *(float2 *)(&c[0]);
            iiiv[1] = *(float2 *)(&c[2]);
            iiiv[2] = *(float2 *)(&c[4]);

            FDTD_FLOAT v[3];
            v[0] = vbase[0];
            v[1] = vbase[1];
            v[2] = vbase[2];

            iiiv[0].y *= (v[2] - vy[2] - v[1] + vz[1]);
            iiiv[1].y *= (v[0] - vz[0] - v[2] + vx[2]);
            iiiv[2].y *= (v[1] - vx[1] - v[0] + vy[0]);

            FDTD_FLOAT* i = curr + offs;
            i[0] = i[0] * iiiv[0].x + iiiv[0].y;
            i[1] = i[1] * iiiv[1].x + iiiv[1].y;
            i[2] = i[2] * iiiv[2].x + iiiv[2].y;
        }
    }
}

__global__ void addInMgKernel(FDTD_FLOAT *p, int local_cell, int n, FDTD_FLOAT val)
{
    p[local_cell * 3 + n] += val;
}

// gather tracked cells (LOCAL flat ids incl. ghost offset) into staging
__global__ void gatherMgKernel(const FDTD_FLOAT *volt, const FDTD_FLOAT *curr,
                               const int *cells, FDTD_FLOAT *stage, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int o = cells[i] * 3;
    FDTD_FLOAT *s = stage + i * 6;
    s[0] = volt[o]; s[1] = volt[o + 1]; s[2] = volt[o + 2];
    s[3] = curr[o]; s[4] = curr[o + 1]; s[5] = curr[o + 2];
}

// per-slab energy over real cells, matching the single-GPU exclusions
// (global x < gnx-1, y < ny-1, z < nz-1); accumulate into *p_sum via atomics.
__global__ void energyMgKernel(const FDTD_FLOAT *volt, const FDTD_FLOAT *curr,
                               double *p_sum, int nx_real, int ny, int nz, bool is_last)
{
    __shared__ double vv[MG_THREADS];
    __shared__ double ii[MG_THREADS];
    int plane = ny * nz;
    int total = nx_real * plane;
    double lv = 0.0, li = 0.0;
    int tid = threadIdx.x;
    for (int cell = blockIdx.x * MG_THREADS + tid; cell < total; cell += gridDim.x * MG_THREADS) {
        int xr = cell / plane;
        int y = (cell / nz) % ny;
        int z = cell % nz;
        if ((!is_last || xr < nx_real - 1) && y < ny - 1 && z < nz - 1) {
            int offs = (cell + plane) * 3;
            const FDTD_FLOAT *v = volt + offs;
            const FDTD_FLOAT *i = curr + offs;
            lv += (double)v[0]*v[0] + (double)v[1]*v[1] + (double)v[2]*v[2];
            li += (double)i[0]*i[0] + (double)i[1]*i[1] + (double)i[2]*i[2];
        }
    }
    vv[tid] = lv; ii[tid] = li;
    __syncthreads();
    for (int s = MG_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) { vv[tid] += vv[tid + s]; ii[tid] += ii[tid + s]; }
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(p_sum, vv[0] * __EPS0__ + ii[0] * __MUE0__);
}

// ---------------------------------------------------------------------------
// per-slab compressed-coefficient build (slice of the operator's host arrays)
// ---------------------------------------------------------------------------
static void build_coeff_slice(
    const FDTD_FLOAT *vv, const FDTD_FLOAT *vi,
    const FDTD_FLOAT *ii, const FDTD_FLOAT *iv,
    int num_cells,
    void **d_index, bool *index_u16, FDTD_FLOAT **d_vv_vi, FDTD_FLOAT **d_ii_iv)
{
    std::vector<unsigned int> index(num_cells);
    std::vector<FDTD_FLOAT> tbl_vvvi, tbl_iiiv;
    std::unordered_map<std::string, unsigned int> lut;

    for (int c = 0; c < num_cells; ++c)
    {
        FDTD_FLOAT sv[12];
        sv[0]  = vv[c*3+0]; sv[1]  = vi[c*3+0];
        sv[2]  = vv[c*3+1]; sv[3]  = vi[c*3+1];
        sv[4]  = vv[c*3+2]; sv[5]  = vi[c*3+2];
        sv[6]  = ii[c*3+0]; sv[7]  = iv[c*3+0];
        sv[8]  = ii[c*3+1]; sv[9]  = iv[c*3+1];
        sv[10] = ii[c*3+2]; sv[11] = iv[c*3+2];

        std::string key((const char*)sv, sizeof(sv));
        auto it = lut.find(key);
        unsigned int idx;
        if (it == lut.end())
        {
            idx = (unsigned int)(tbl_vvvi.size() / 6);
            for (int j = 0; j < 6; ++j) tbl_vvvi.push_back(sv[j]);
            for (int j = 0; j < 6; ++j) tbl_iiiv.push_back(sv[6 + j]);
            lut.emplace(std::move(key), idx);
        }
        else
            idx = it->second;
        index[c] = idx;
    }

    unsigned int uniq = (unsigned int)(tbl_vvvi.size() / 6);
    *index_u16 = (uniq <= 65535u);
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

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

Engine_cuda_mgpu* Engine_cuda_mgpu::New(const Operator_CUDA* op, int num_slabs,
                                        bool virtual_mode, unsigned int base_device)
{
    std::cout << "Create CUDA multi-GPU FDTD engine: " << num_slabs << " slab(s)"
              << (virtual_mode ? " [VIRTUAL: all on one device]" : "") << std::endl;
    Engine_cuda_mgpu* e = new Engine_cuda_mgpu(op, num_slabs, virtual_mode);
    e->setCUDAdevice(base_device);
    e->Init();
    return e;
}

Engine_cuda_mgpu::Engine_cuda_mgpu(const Operator_CUDA* op, int num_slabs, bool virtual_mode)
    : Engine_cuda(op)
{
    m_num_slabs = num_slabs;
    m_virtual = virtual_mode;
}

Engine_cuda_mgpu::~Engine_cuda_mgpu()
{
    Reset();
}

void Engine_cuda_mgpu::Init()
{
    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    if (nDevices <= 0)
        throw std::runtime_error("NO CUDA device found");

    m_dim = dim3(numLines[0], numLines[1], numLines[2]);
    numTS = 0;

    int gnx = numLines[0], ny = numLines[1], nz = numLines[2];
    size_t planeF = (size_t)ny * nz * 3;

    // balanced contiguous slabs along x
    m_ctx.resize(m_num_slabs);
    int base = gnx / m_num_slabs, rem = gnx % m_num_slabs, xcur = 0;
    for (int g = 0; g < m_num_slabs; ++g)
    {
        int nx = base + (g < rem ? 1 : 0);
        m_ctx[g].x_start = xcur;
        m_ctx[g].x_end   = xcur + nx;
        xcur += nx;
        m_ctx[g].device = m_virtual ? (int)m_cuda_device_number : g;
        if (!m_virtual && m_ctx[g].device >= nDevices)
            throw std::runtime_error("not enough CUDA devices for requested slab count");
    }

    printf("mgpu: %dx%dx%d cells over %d slab(s):", gnx, ny, nz, m_num_slabs);
    for (int g = 0; g < m_num_slabs; ++g)
        printf(" [%d,%d)@dev%d", m_ctx[g].x_start, m_ctx[g].x_end, m_ctx[g].device);
    printf("\n"); fflush(stdout);

    // peer access between adjacent slabs (no-op in virtual mode)
    if (!m_virtual)
        for (int g = 0; g + 1 < m_num_slabs; ++g)
        {
            int a = m_ctx[g].device, b = m_ctx[g + 1].device;
            int can = 0;
            cudaDeviceCanAccessPeer(&can, a, b);
            if (can) { cudaSetDevice(a); cudaDeviceEnablePeerAccess(b, 0); cudaGetLastError(); }
            cudaDeviceCanAccessPeer(&can, b, a);
            if (can) { cudaSetDevice(b); cudaDeviceEnablePeerAccess(a, 0); cudaGetLastError(); }
        }

    m_d_idx.resize(m_num_slabs);   m_idx_u16.resize(m_num_slabs);
    m_d_vvvi.resize(m_num_slabs);  m_d_iiiv.resize(m_num_slabs);
    m_d_energy.resize(m_num_slabs);
    m_ev_volt[0].resize(m_num_slabs); m_ev_volt[1].resize(m_num_slabs);
    m_ev_curr[0].resize(m_num_slabs); m_ev_curr[1].resize(m_num_slabs);
    m_cell_list.resize(m_num_slabs); m_cell_glob.resize(m_num_slabs);
    m_d_cells.assign(m_num_slabs, NULL);
    m_d_stage.assign(m_num_slabs, NULL);
    m_h_stage.assign(m_num_slabs, NULL);
    m_stage_cap.assign(m_num_slabs, 0);

    for (int g = 0; g < m_num_slabs; ++g)
    {
        CudaSlabCtx &c = m_ctx[g];
        int nx_real = c.x_end - c.x_start;
        c.local_dim = dim3(nx_real + 2, ny, nz);

        checkCuda(cudaSetDevice(c.device));
        checkCuda(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking));

        size_t bytes = (size_t)(nx_real + 2) * planeF * sizeof(FDTD_FLOAT);
        checkCuda(cudaMalloc(&c.d_volt, bytes));
        checkCuda(cudaMalloc(&c.d_curr, bytes));
        checkCuda(cudaMemset(c.d_volt, 0, bytes));
        checkCuda(cudaMemset(c.d_curr, 0, bytes));

        checkCuda(cudaMalloc(&c.d_numTS, sizeof(int)));
        checkCuda(cudaMemset(c.d_numTS, 0, sizeof(int)));

        // coefficient slice for this slab's real cells
        size_t cellOfs = (size_t)c.x_start * ny * nz * 3;
        bool u16 = false;
        build_coeff_slice(Op->vv_ptr->data() + cellOfs, Op->vi_ptr->data() + cellOfs,
                          Op->ii_ptr->data() + cellOfs, Op->iv_ptr->data() + cellOfs,
                          nx_real * ny * nz,
                          &m_d_idx[g], &u16, &m_d_vvvi[g], &m_d_iiiv[g]);
        m_idx_u16[g] = u16;

        checkCuda(cudaMalloc(&m_d_energy[g], sizeof(double)));

        for (int p = 0; p < 2; ++p)
        {
            checkCuda(cudaEventCreateWithFlags(&m_ev_volt[p][g], cudaEventDisableTiming));
            checkCuda(cudaEventCreateWithFlags(&m_ev_curr[p][g], cudaEventDisableTiming));
        }
    }

    // pinned full-size host mirror + per-slab pinned energy results
    size_t mirrorBytes = (size_t)gnx * planeF * sizeof(FDTD_FLOAT);
    checkCuda(cudaHostAlloc(&m_h_volt, mirrorBytes, cudaHostAllocDefault));
    checkCuda(cudaHostAlloc(&m_h_curr, mirrorBytes, cudaHostAllocDefault));
    memset(m_h_volt, 0, mirrorBytes);
    memset(m_h_curr, 0, mirrorBytes);
    checkCuda(cudaHostAlloc(&m_h_energy, m_num_slabs * sizeof(double), cudaHostAllocDefault));

    InitExtensions();
    SortExtensionByPriority();

    for (size_t n = 0; n < m_Eng_exts.size(); ++n)
        if (!m_Eng_exts.at(n)->IsCUDACapable())
            std::cerr << "openEMS CUDA mgpu WARNING: extension '"
                      << m_Eng_exts.at(n)->GetExtensionName()
                      << "' has no CUDA implementation -- its effect is silently ignored."
                      << std::endl;
}

void Engine_cuda_mgpu::Reset()
{
    for (size_t g = 0; g < m_ctx.size(); ++g)
    {
        CudaSlabCtx &c = m_ctx[g];
        cudaSetDevice(c.device);
        if (c.d_volt)  cudaFree(c.d_volt);
        if (c.d_curr)  cudaFree(c.d_curr);
        if (c.d_numTS) cudaFree(c.d_numTS);
        if (m_d_idx[g])    cudaFree(m_d_idx[g]);
        if (m_d_vvvi[g])   cudaFree(m_d_vvvi[g]);
        if (m_d_iiiv[g])   cudaFree(m_d_iiiv[g]);
        if (m_d_energy[g]) cudaFree(m_d_energy[g]);
        if (m_d_cells[g])  cudaFree(m_d_cells[g]);
        if (m_d_stage[g])  cudaFree(m_d_stage[g]);
        if (m_h_stage[g])  cudaFreeHost(m_h_stage[g]);
        for (int p = 0; p < 2; ++p)
        {
            if (m_ev_volt[p][g]) cudaEventDestroy(m_ev_volt[p][g]);
            if (m_ev_curr[p][g]) cudaEventDestroy(m_ev_curr[p][g]);
        }
        if (c.stream) cudaStreamDestroy(c.stream);
    }
    m_ctx.clear();
    m_d_idx.clear(); m_d_vvvi.clear(); m_d_iiiv.clear(); m_d_energy.clear();
    m_d_cells.clear(); m_d_stage.clear(); m_h_stage.clear(); m_stage_cap.clear();
    for (int p = 0; p < 2; ++p) { m_ev_volt[p].clear(); m_ev_curr[p].clear(); }
    if (m_h_volt)   { cudaFreeHost(m_h_volt);   m_h_volt = NULL; }
    if (m_h_curr)   { cudaFreeHost(m_h_curr);   m_h_curr = NULL; }
    if (m_h_energy) { cudaFreeHost(m_h_energy); m_h_energy = NULL; }
    ClearExtensions();
}

// loud failure for any non-mgpu-aware device access
FDTD_FLOAT* Engine_cuda_mgpu::GetDeviceVoltData()
{
    fprintf(stderr, "Engine_cuda_mgpu: GetDeviceVoltData() called by a non-mgpu-aware extension -- unsupported.\n");
    abort();
}
FDTD_FLOAT* Engine_cuda_mgpu::GetDeviceCurrData()
{
    fprintf(stderr, "Engine_cuda_mgpu: GetDeviceCurrData() called by a non-mgpu-aware extension -- unsupported.\n");
    abort();
}
int* Engine_cuda_mgpu::GetDeviceNumTS()
{
    fprintf(stderr, "Engine_cuda_mgpu: GetDeviceNumTS() called by a non-mgpu-aware extension -- unsupported.\n");
    abort();
}

int Engine_cuda_mgpu::OwnerSlab(int x) const
{
    for (int g = 0; g < m_num_slabs; ++g)
        if (x < m_ctx[g].x_end) return g;
    return m_num_slabs - 1;
}

// ---------------------------------------------------------------------------
// timestep
// ---------------------------------------------------------------------------

void Engine_cuda_mgpu::UpdateVoltages(unsigned int startX, unsigned int numX)
{
    (void)startX; (void)numX;
    int ny = numLines[1], nz = numLines[2];
    for (int g = 0; g < m_num_slabs; ++g)
    {
        CudaSlabCtx &c = m_ctx[g];
        cudaSetDevice(c.device);
        int N = (c.x_end - c.x_start) * ny * nz;
        int blocks = (N + MG_THREADS - 1) / MG_THREADS;
        if (m_idx_u16[g])
            updateVoltagesMgKernel<<<blocks, MG_THREADS, 0, c.stream>>>(
                c.d_volt, c.d_curr, (const unsigned short*)m_d_idx[g], m_d_vvvi[g],
                N, ny, nz, g == 0);
        else
            updateVoltagesMgKernel<<<blocks, MG_THREADS, 0, c.stream>>>(
                c.d_volt, c.d_curr, (const unsigned int*)m_d_idx[g], m_d_vvvi[g],
                N, ny, nz, g == 0);
    }
}

void Engine_cuda_mgpu::UpdateCurrents(unsigned int startX, unsigned int numX)
{
    (void)startX; (void)numX;
    int ny = numLines[1], nz = numLines[2];
    for (int g = 0; g < m_num_slabs; ++g)
    {
        CudaSlabCtx &c = m_ctx[g];
        cudaSetDevice(c.device);
        bool is_last = (g == m_num_slabs - 1);
        int nplanes = (c.x_end - c.x_start) - (is_last ? 1 : 0);
        int N = nplanes * ny * nz;
        if (N <= 0) continue;
        int blocks = (N + MG_THREADS - 1) / MG_THREADS;
        if (m_idx_u16[g])
            updateCurrentsMgKernel<<<blocks, MG_THREADS, 0, c.stream>>>(
                c.d_curr, c.d_volt, (const unsigned short*)m_d_idx[g], m_d_iiiv[g], N, ny, nz);
        else
            updateCurrentsMgKernel<<<blocks, MG_THREADS, 0, c.stream>>>(
                c.d_curr, c.d_volt, (const unsigned int*)m_d_idx[g], m_d_iiiv[g], N, ny, nz);
    }
}

void Engine_cuda_mgpu::HaloSendVolt(int parity)
{
    // slab g's FIRST real volt plane -> slab g-1's right ghost
    size_t planeF = (size_t)numLines[1] * numLines[2] * 3;
    size_t bytes = planeF * sizeof(FDTD_FLOAT);
    for (int g = 1; g < m_num_slabs; ++g)
    {
        CudaSlabCtx &src = m_ctx[g], &dst = m_ctx[g - 1];
        int dst_nx = dst.x_end - dst.x_start;
        FDTD_FLOAT *dp = dst.d_volt + (size_t)(dst_nx + 1) * planeF;
        FDTD_FLOAT *sp = src.d_volt + planeF;
        cudaSetDevice(src.device);
        if (src.device == dst.device)
            checkCuda(cudaMemcpyAsync(dp, sp, bytes, cudaMemcpyDeviceToDevice, src.stream));
        else
            checkCuda(cudaMemcpyPeerAsync(dp, dst.device, sp, src.device, bytes, src.stream));
        checkCuda(cudaEventRecord(m_ev_volt[parity][g], src.stream));
    }
}

void Engine_cuda_mgpu::HaloSendCurr(int parity)
{
    // slab g's LAST real curr plane -> slab g+1's left ghost
    size_t planeF = (size_t)numLines[1] * numLines[2] * 3;
    size_t bytes = planeF * sizeof(FDTD_FLOAT);
    for (int g = 0; g + 1 < m_num_slabs; ++g)
    {
        CudaSlabCtx &src = m_ctx[g], &dst = m_ctx[g + 1];
        int src_nx = src.x_end - src.x_start;
        FDTD_FLOAT *sp = src.d_curr + (size_t)src_nx * planeF;   // last real plane
        FDTD_FLOAT *dp = dst.d_curr;                              // left ghost
        cudaSetDevice(src.device);
        if (src.device == dst.device)
            checkCuda(cudaMemcpyAsync(dp, sp, bytes, cudaMemcpyDeviceToDevice, src.stream));
        else
            checkCuda(cudaMemcpyPeerAsync(dp, dst.device, sp, src.device, bytes, src.stream));
        checkCuda(cudaEventRecord(m_ev_curr[parity][g], src.stream));
    }
}

void Engine_cuda_mgpu::RunOneTimestepMg(int parity)
{
    int prev = parity ^ 1;

    // wait for the curr halo produced by the left neighbour LAST timestep
    for (int g = 1; g < m_num_slabs; ++g)
    {
        cudaSetDevice(m_ctx[g].device);
        checkCuda(cudaStreamWaitEvent(m_ctx[g].stream, m_ev_curr[prev][g - 1], 0));
    }

    DoPreVoltageUpdates();
    UpdateVoltages(0, numLines[0]);
    DoPostVoltageUpdates();
    Apply2Voltages();

    HaloSendVolt(parity);
    for (int g = 0; g + 1 < m_num_slabs; ++g)
    {
        cudaSetDevice(m_ctx[g].device);
        checkCuda(cudaStreamWaitEvent(m_ctx[g].stream, m_ev_volt[parity][g + 1], 0));
    }

    DoPreCurrentUpdates();
    UpdateCurrents(0, numLines[0] - 1);
    DoPostCurrentUpdates();
    Apply2Current();

    HaloSendCurr(parity);

    for (int g = 0; g < m_num_slabs; ++g)
    {
        cudaSetDevice(m_ctx[g].device);
        incrementNumTSKernel<<<1, 1, 0, m_ctx[g].stream>>>(m_ctx[g].d_numTS);
    }
}

bool Engine_cuda_mgpu::IterateTS(unsigned int iterTS)
{
    if (m_volt_dirty || m_curr_dirty)
        UploadHostMirror();
    m_locked = true;

    // seed per-slab device timestep counters for this chunk
    for (int g = 0; g < m_num_slabs; ++g)
    {
        cudaSetDevice(m_ctx[g].device);
        setNumTSKernel<<<1, 1, 0, m_ctx[g].stream>>>(m_ctx[g].d_numTS, (int)numTS);
    }

    for (unsigned int iter = 0; iter < iterTS; ++iter)
    {
        RunOneTimestepMg((int)(m_step_counter & 1));
        ++m_step_counter;
    }
    numTS += iterTS;

    // refresh the host mirror for probes/dumps
    if (m_full)
    {
        size_t planeF = (size_t)numLines[1] * numLines[2] * 3;
        for (int g = 0; g < m_num_slabs; ++g)
        {
            CudaSlabCtx &c = m_ctx[g];
            cudaSetDevice(c.device);
            size_t n = (size_t)(c.x_end - c.x_start) * planeF;
            checkCuda(cudaMemcpyAsync(m_h_volt + (size_t)c.x_start * planeF, c.d_volt + planeF,
                                      n * sizeof(FDTD_FLOAT), cudaMemcpyDeviceToHost, c.stream));
            checkCuda(cudaMemcpyAsync(m_h_curr + (size_t)c.x_start * planeF, c.d_curr + planeF,
                                      n * sizeof(FDTD_FLOAT), cudaMemcpyDeviceToHost, c.stream));
        }
    }
    else
        SelectiveReadbackMg();

    for (int g = 0; g < m_num_slabs; ++g)
    {
        cudaSetDevice(m_ctx[g].device);
        checkCuda(cudaStreamSynchronize(m_ctx[g].stream));
    }
    SelectiveReadbackFinishMg();

    m_locked = false;
    m_volt_dirty = 0;
    m_curr_dirty = 0;
    return true;
}

// ---------------------------------------------------------------------------
// host mirror + probes
// ---------------------------------------------------------------------------

void Engine_cuda_mgpu::UploadHostMirror()
{
    size_t planeF = (size_t)numLines[1] * numLines[2] * 3;
    for (int g = 0; g < m_num_slabs; ++g)
    {
        CudaSlabCtx &c = m_ctx[g];
        cudaSetDevice(c.device);
        size_t n = (size_t)(c.x_end - c.x_start) * planeF;
        checkCuda(cudaMemcpy(c.d_volt + planeF, m_h_volt + (size_t)c.x_start * planeF,
                             n * sizeof(FDTD_FLOAT), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(c.d_curr + planeF, m_h_curr + (size_t)c.x_start * planeF,
                             n * sizeof(FDTD_FLOAT), cudaMemcpyHostToDevice));
        // refresh ghost planes from the mirror too (they are otherwise stale
        // until the next halo exchange, but the first kernels read them first)
        if (g > 0)
            checkCuda(cudaMemcpy(c.d_curr, m_h_curr + (size_t)(c.x_start - 1) * planeF,
                                 planeF * sizeof(FDTD_FLOAT), cudaMemcpyHostToDevice));
        if (g + 1 < m_num_slabs)
        {
            int nx = c.x_end - c.x_start;
            checkCuda(cudaMemcpy(c.d_volt + (size_t)(nx + 1) * planeF,
                                 m_h_volt + (size_t)c.x_end * planeF,
                                 planeF * sizeof(FDTD_FLOAT), cudaMemcpyHostToDevice));
        }
    }
}

void Engine_cuda_mgpu::EnsureCellHostMg(int cell) const
{
    if (m_full) return;
    if (m_cells.find(cell) != m_cells.end()) return;
    m_cells.insert(cell);
    m_dirty = true;

    int ny = numLines[1], nz = numLines[2];
    int plane = ny * nz;
    int x = cell / plane, rem = cell % plane;
    int g = OwnerSlab(x);
    const CudaSlabCtx &c = m_ctx[g];
    size_t lofs = ((size_t)(x - c.x_start + 1) * plane + rem) * 3;

    cudaMemcpy(m_h_volt + (size_t)cell * 3, c.d_volt + lofs, 3 * sizeof(FDTD_FLOAT), cudaMemcpyDeviceToHost);
    cudaMemcpy(m_h_curr + (size_t)cell * 3, c.d_curr + lofs, 3 * sizeof(FDTD_FLOAT), cudaMemcpyDeviceToHost);

    size_t num_cells = (size_t)numLines[0] * plane;
    if (m_cells.size() > num_cells / 20)
        m_full = true;
}

void Engine_cuda_mgpu::SelectiveReadbackMg()
{
    if (m_cells.empty()) return;

    int ny = numLines[1], nz = numLines[2], plane = ny * nz;
    if (m_dirty)
    {
        for (int g = 0; g < m_num_slabs; ++g) { m_cell_list[g].clear(); m_cell_glob[g].clear(); }
        for (int cell : m_cells)
        {
            int x = cell / plane, rem = cell % plane;
            int g = OwnerSlab(x);
            m_cell_list[g].push_back((int)(((size_t)(x - m_ctx[g].x_start + 1) * plane + rem)));
            m_cell_glob[g].push_back(cell);
        }
        for (int g = 0; g < m_num_slabs; ++g)
        {
            size_t n = m_cell_list[g].size();
            if (n == 0) continue;
            cudaSetDevice(m_ctx[g].device);
            if (n > m_stage_cap[g])
            {
                if (m_d_cells[g]) cudaFree(m_d_cells[g]);
                if (m_d_stage[g]) cudaFree(m_d_stage[g]);
                if (m_h_stage[g]) cudaFreeHost(m_h_stage[g]);
                m_stage_cap[g] = n * 2;
                checkCuda(cudaMalloc(&m_d_cells[g], m_stage_cap[g] * sizeof(int)));
                checkCuda(cudaMalloc(&m_d_stage[g], m_stage_cap[g] * 6 * sizeof(FDTD_FLOAT)));
                checkCuda(cudaHostAlloc(&m_h_stage[g], m_stage_cap[g] * 6 * sizeof(FDTD_FLOAT), cudaHostAllocDefault));
            }
            checkCuda(cudaMemcpyAsync(m_d_cells[g], m_cell_list[g].data(), n * sizeof(int),
                                      cudaMemcpyHostToDevice, m_ctx[g].stream));
        }
        m_dirty = false;
    }

    for (int g = 0; g < m_num_slabs; ++g)
    {
        int n = (int)m_cell_list[g].size();
        if (n == 0) continue;
        cudaSetDevice(m_ctx[g].device);
        int blocks = (n + 127) / 128;
        gatherMgKernel<<<blocks, 128, 0, m_ctx[g].stream>>>(m_ctx[g].d_volt, m_ctx[g].d_curr,
                                                            m_d_cells[g], m_d_stage[g], n);
        checkCuda(cudaMemcpyAsync(m_h_stage[g], m_d_stage[g], (size_t)n * 6 * sizeof(FDTD_FLOAT),
                                  cudaMemcpyDeviceToHost, m_ctx[g].stream));
    }
    m_pending = true;
}

void Engine_cuda_mgpu::SelectiveReadbackFinishMg()
{
    if (!m_pending) return;
    m_pending = false;
    for (int g = 0; g < m_num_slabs; ++g)
    {
        int n = (int)m_cell_glob[g].size();
        for (int i = 0; i < n; ++i)
        {
            size_t o = (size_t)m_cell_glob[g][i] * 3;
            const FDTD_FLOAT *s = m_h_stage[g] + (size_t)i * 6;
            m_h_volt[o] = s[0]; m_h_volt[o + 1] = s[1]; m_h_volt[o + 2] = s[2];
            m_h_curr[o] = s[3]; m_h_curr[o + 1] = s[4]; m_h_curr[o + 2] = s[5];
        }
    }
}

FDTD_FLOAT Engine_cuda_mgpu::GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
{
    assert(!m_locked);
    EnsureCellHostMg(FlatIndex(x, y, z));
    return m_h_volt[(size_t)FlatIndex(x, y, z) * 3 + n];
}
FDTD_FLOAT Engine_cuda_mgpu::GetVolt(unsigned int n, const unsigned int pos[3]) const
{ return GetVolt(n, pos[0], pos[1], pos[2]); }

FDTD_FLOAT Engine_cuda_mgpu::GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
{
    assert(!m_locked);
    EnsureCellHostMg(FlatIndex(x, y, z));
    return m_h_curr[(size_t)FlatIndex(x, y, z) * 3 + n];
}
FDTD_FLOAT Engine_cuda_mgpu::GetCurr(unsigned int n, const unsigned int pos[3]) const
{ return GetCurr(n, pos[0], pos[1], pos[2]); }

void Engine_cuda_mgpu::SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val)
{
    assert(!m_locked);
    m_h_volt[(size_t)FlatIndex(x, y, z) * 3 + n] = val;
    m_volt_dirty += 1;
}
void Engine_cuda_mgpu::SetVolt(unsigned int n, const unsigned int pos[3], FDTD_FLOAT val)
{ SetVolt(n, pos[0], pos[1], pos[2], val); }

void Engine_cuda_mgpu::SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val)
{
    assert(!m_locked);
    m_h_curr[(size_t)FlatIndex(x, y, z) * 3 + n] = val;
    m_curr_dirty += 1;
}
void Engine_cuda_mgpu::SetCurr(unsigned int n, const unsigned int pos[3], FDTD_FLOAT val)
{ SetCurr(n, pos[0], pos[1], pos[2], val); }

void Engine_cuda_mgpu::AddVolt(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value)
{
    int plane = numLines[1] * numLines[2];
    int g = OwnerSlab((int)pos[0]);
    const CudaSlabCtx &c = m_ctx[g];
    int lcell = (int)(((size_t)(pos[0] - c.x_start + 1) * plane
                       + pos[1] * numLines[2] + pos[2]));
    cudaSetDevice(c.device);
    addInMgKernel<<<1, 1, 0, c.stream>>>(c.d_volt, lcell, (int)n, value);
}

void Engine_cuda_mgpu::AddCurr(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value)
{
    int plane = numLines[1] * numLines[2];
    int g = OwnerSlab((int)pos[0]);
    const CudaSlabCtx &c = m_ctx[g];
    int lcell = (int)(((size_t)(pos[0] - c.x_start + 1) * plane
                       + pos[1] * numLines[2] + pos[2]));
    cudaSetDevice(c.device);
    addInMgKernel<<<1, 1, 0, c.stream>>>(c.d_curr, lcell, (int)n, value);
}

double Engine_cuda_mgpu::CalcFastEnergy()
{
    int ny = numLines[1], nz = numLines[2];
    for (int g = 0; g < m_num_slabs; ++g)
    {
        CudaSlabCtx &c = m_ctx[g];
        cudaSetDevice(c.device);
        checkCuda(cudaMemsetAsync(m_d_energy[g], 0, sizeof(double), c.stream));
        int nx_real = c.x_end - c.x_start;
        int total = nx_real * ny * nz;
        int blocks = (total + MG_THREADS - 1) / MG_THREADS;
        if (blocks > 184) blocks = 184;
        energyMgKernel<<<blocks, MG_THREADS, 0, c.stream>>>(c.d_volt, c.d_curr, m_d_energy[g],
                                                            nx_real, ny, nz, g == m_num_slabs - 1);
        checkCuda(cudaMemcpyAsync(&m_h_energy[g], m_d_energy[g], sizeof(double),
                                  cudaMemcpyDeviceToHost, c.stream));
    }
    double sum = 0.0;
    for (int g = 0; g < m_num_slabs; ++g)
    {
        cudaSetDevice(m_ctx[g].device);
        checkCuda(cudaStreamSynchronize(m_ctx[g].stream));
        sum += m_h_energy[g];
    }
    return sum;
}
