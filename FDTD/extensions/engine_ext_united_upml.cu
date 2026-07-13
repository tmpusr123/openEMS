
#include "engine_ext_united_upml.h"
#include "engine_ext_excitation.h"
#include "operator_ext_upml.h"
#include "FDTD/engine_cuda_mgpu.h"
#include "tools/arraylib/array_nijk.h"



#include <cuda_runtime.h>
#include <unordered_map>
#include <string>
#include <vector>
#include <algorithm>
#include "hemi/hemi_error.h"
#include "hemi/grid_stride_range.h"



// ---- multi-GPU: clip every PML block to each slab's x-range and upload per
// device. Block starts are translated to slab-local coordinates (x includes
// the +1 ghost-plane offset), so the existing kernels work unmodified with the
// slab's local dim. Flux state is owned per slab; coefficients dedup per slab.
void Engine_Ext_United_UPML::SetEngineMg(Engine_cuda_mgpu* mg)
{
    m_mgpu = true;
    int nslab = mg->NumSlabs();
    mg_blks.assign(nslab, NULL);
    mg_nblocks.assign(nslab, 0);
    mg_maxcells.assign(nslab, 0);

    for (int g = 0; g < nslab; ++g)
    {
        const CudaSlabCtx &c = mg->Slab(g);
        checkCuda(cudaSetDevice(c.device));

        std::unordered_map<std::string, unsigned short> uniq;
        std::vector<float> table;
        std::vector<upml_block_t> blocks;
        std::vector<std::vector<unsigned short> > host_index;

        for (size_t b = 0; b < m_Op_UPML_List->size(); ++b)
        {
            Operator_Ext_UPML *op = m_Op_UPML_List->at(b);
            int bs = op->m_StartPos[0];
            int be = bs + op->m_numLines[0];
            int ly = op->m_numLines[1], lz = op->m_numLines[2];
            int cs = std::max(bs, c.x_start);
            int ce = std::min(be, c.x_end);
            if (cs >= ce) continue;   // block does not intersect this slab

            int lx = ce - cs;
            int blk_cells = lx * ly * lz;
            int n = blk_cells * 3;
            int orig_off = (cs - bs) * ly * lz * 3;   // offset into the op's host arrays

            const FDTD_FLOAT *vv = op->vv.data(),  *vvfn = op->vvfn.data(), *vvfo = op->vvfo.data();
            const FDTD_FLOAT *ii = op->ii.data(),  *iifn = op->iifn.data(), *iifo = op->iifo.data();

            host_index.push_back(std::vector<unsigned short>(n));
            std::vector<unsigned short> &hidx = host_index.back();
            for (int i = 0; i < n; i++)
            {
                int io = i + orig_off;
                float sext[6] = { vv[io], vvfn[io], vvfo[io], ii[io], iifn[io], iifo[io] };
                std::string key((const char*)sext, sizeof(sext));
                auto it = uniq.find(key);
                unsigned short idx;
                if (it == uniq.end())
                {
                    idx = (unsigned short)(table.size() / 6);
                    uniq.emplace(key, idx);
                    for (int k = 0; k < 6; k++) table.push_back(sext[k]);
                }
                else
                    idx = it->second;
                hidx[i] = idx;
            }

            upml_block_t blk;
            blk.start = dim3(cs - c.x_start + 1, op->m_StartPos[1], op->m_StartPos[2]);
            blk.lines = dim3(lx, ly, lz);

            FDTD_FLOAT *vf, *cf;
            checkCuda(cudaMalloc(&vf, (size_t)n * sizeof(FDTD_FLOAT)));
            checkCuda(cudaMalloc(&cf, (size_t)n * sizeof(FDTD_FLOAT)));
            checkCuda(cudaMemset(vf, 0, (size_t)n * sizeof(FDTD_FLOAT)));
            checkCuda(cudaMemset(cf, 0, (size_t)n * sizeof(FDTD_FLOAT)));
            blk.volt_flux = vf; blk.curr_flux = cf;
            mg_allocs.push_back(std::make_pair(c.device, (void*)vf));
            mg_allocs.push_back(std::make_pair(c.device, (void*)cf));

            blocks.push_back(blk);
            if (blk_cells > mg_maxcells[g]) mg_maxcells[g] = blk_cells;
        }

        if (blocks.empty()) continue;
        if (table.size() / 6 > 65535)
            fprintf(stderr, "United UPML mgpu WARNING: unique PML coeffs exceed uint16 index space.\n");

        FDTD_FLOAT *d_table;
        checkCuda(cudaMalloc(&d_table, table.size() * sizeof(float)));
        checkCuda(cudaMemcpy(d_table, table.data(), table.size() * sizeof(float), cudaMemcpyHostToDevice));
        mg_allocs.push_back(std::make_pair(c.device, (void*)d_table));

        for (size_t b = 0; b < blocks.size(); ++b)
        {
            unsigned short *d_idx;
            size_t nbytes = host_index[b].size() * sizeof(unsigned short);
            checkCuda(cudaMalloc(&d_idx, nbytes));
            checkCuda(cudaMemcpy(d_idx, host_index[b].data(), nbytes, cudaMemcpyHostToDevice));
            mg_allocs.push_back(std::make_pair(c.device, (void*)d_idx));
            blocks[b].coeff_index = d_idx;
            blocks[b].coeff_table = d_table;
        }

        upml_block_t *d_blks;
        checkCuda(cudaMalloc(&d_blks, blocks.size() * sizeof(upml_block_t)));
        checkCuda(cudaMemcpy(d_blks, blocks.data(), blocks.size() * sizeof(upml_block_t), cudaMemcpyHostToDevice));
        mg_allocs.push_back(std::make_pair(c.device, (void*)d_blks));
        mg_blks[g] = d_blks;
        mg_nblocks[g] = (int)blocks.size();
    }
}

void Engine_Ext_United_UPML::SetEngine(Engine* eng)
{
    m_Eng = eng;
    if (eng->GetType() != Engine::CUDA) {
        return;
    }

    if (Engine_cuda_mgpu* mg = dynamic_cast<Engine_cuda_mgpu*>(eng)) {
        SetEngineMg(mg);
        return;
    }

    int upml_blocks = m_Op_UPML_List->size();

    checkCuda(cudaMalloc(&m_d_blks, sizeof(upml_block_t) * upml_blocks));

    m_num_of_cells_in_block = new int[upml_blocks];
    m_max_num_cells_in_block = 0;

    // Deduplicate the 6 per-(cell,component) PML coefficients into a shared table
    // (they vary only with depth into the absorber, so this dedups to a few
    // hundred unique sextets). Sextet layout matches upml_block_t's kernels:
    // [vv, vvfn, vvfo, ii, iifn, iifo].
    std::unordered_map<std::string, unsigned short> uniq;
    std::vector<float> table;                                  // 6 floats / unique
    std::vector<std::vector<unsigned short>> host_index(upml_blocks);
    std::vector<upml_block_t> blocks(upml_blocks);

    for (int b = 0; b < upml_blocks; b++)
    {
        Operator_Ext_UPML *op = m_Op_UPML_List->at(b);
        int blk_size = op->m_numLines[0] * op->m_numLines[1] * op->m_numLines[2];

        m_num_of_cells_in_block[b] = blk_size;
        if (blk_size > m_max_num_cells_in_block)
            m_max_num_cells_in_block = blk_size;

        // volt and curr flux are per-(cell,comp) state, device-only, zeroed
        CudaHelper::Array<FDTD_FLOAT> *volt_flux = new CudaHelper::Array<FDTD_FLOAT>(blk_size * 3, NULL);
        CudaHelper::Array<FDTD_FLOAT> *curr_flux = new CudaHelper::Array<FDTD_FLOAT>(blk_size * 3, NULL);
        m_volt_fluxes.push_back(volt_flux);
        m_curr_fluxes.push_back(curr_flux);
        volt_flux->clear();
        curr_flux->clear();

        const FDTD_FLOAT *vv = op->vv.data(),  *vvfn = op->vvfn.data(), *vvfo = op->vvfo.data();
        const FDTD_FLOAT *ii = op->ii.data(),  *iifn = op->iifn.data(), *iifo = op->iifo.data();

        int n = blk_size * 3;
        host_index[b].resize(n);
        for (int i = 0; i < n; i++)
        {
            float sext[6] = { vv[i], vvfn[i], vvfo[i], ii[i], iifn[i], iifo[i] };
            std::string key((const char*)sext, sizeof(sext));
            auto it = uniq.find(key);
            unsigned short idx;
            if (it == uniq.end())
            {
                idx = (unsigned short)(table.size() / 6);
                uniq.emplace(key, idx);
                for (int k = 0; k < 6; k++) table.push_back(sext[k]);
            }
            else
                idx = it->second;
            host_index[b][i] = idx;
        }

        blocks[b].start = dim3(op->m_StartPos[0], op->m_StartPos[1], op->m_StartPos[2]);
        blocks[b].lines = dim3(op->m_numLines[0], op->m_numLines[1], op->m_numLines[2]);
        blocks[b].volt_flux = volt_flux->device_data();
        blocks[b].curr_flux = curr_flux->device_data();
    }

    size_t num_unique = table.size() / 6;
    // PML unique count is tiny in practice (~hundreds); guard the uint16 space.
    if (num_unique > 65535)
        fprintf(stderr, "United UPML WARNING: %zu unique PML coeffs exceeds uint16 index space -- "
                        "compression indices will wrap; use CPU engine for this model.\n", num_unique);

    // Upload the shared table once, then each block's uint16 index array.
    checkCuda(cudaMalloc(&m_d_coeff_table, table.size() * sizeof(float)));
    checkCuda(cudaMemcpy(m_d_coeff_table, table.data(), table.size() * sizeof(float), cudaMemcpyHostToDevice));

    m_d_coeff_index.assign(upml_blocks, NULL);
    for (int b = 0; b < upml_blocks; b++)
    {
        size_t nbytes = host_index[b].size() * sizeof(unsigned short);
        checkCuda(cudaMalloc(&m_d_coeff_index[b], nbytes));
        checkCuda(cudaMemcpy(m_d_coeff_index[b], host_index[b].data(), nbytes, cudaMemcpyHostToDevice));
        blocks[b].coeff_index = m_d_coeff_index[b];
        blocks[b].coeff_table = m_d_coeff_table;
        checkCuda(cudaMemcpy(m_d_blks + b, &blocks[b], sizeof(upml_block_t), cudaMemcpyHostToDevice));
    }
}

__device__
int flat_indx_in_full_grid(int i, const upml_block_t *area, const dim3 *dim)
{
    int loc_x = i / (area->lines.y * area->lines.z);
    int loc_y = (i / area->lines.z) % area->lines.y;
    int loc_z = i % area->lines.z;

    int x = loc_x + area->start.x;
    int y = loc_y + area->start.y;
    int z = loc_z + area->start.z;  

    return (x * dim->y * dim->z + y * dim->z + z);
}

__global__
void PreVoltageUpdateKernel(FDTD_FLOAT *d_volt, const upml_block_t *ublk, const dim3 dim)
{
    // Flat 1-D indexing over the block's flux entries (i = cell*3 + component),
    // identical layout/logic to the original dim3(341,3) version but with
    // consecutive threads hitting consecutive i -> coalesced loads on the
    // block-local flux/vv/vvfo arrays (the dim3(341,3) form did stride-3 loads).
    const upml_block_t area = ublk[blockIdx.y];
    int N = area.lines.x * area.lines.y * area.lines.z * 3;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int gi = flat_indx_in_full_grid(i / 3, &area, &dim) * 3 + (i % 3);

        const FDTD_FLOAT *c = area.coeff_table + (int)area.coeff_index[i] * 6;
        FDTD_FLOAT *volt = d_volt + gi;
        FDTD_FLOAT *flux = area.volt_flux + i;

        FDTD_FLOAT f = c[0] * volt[0] - c[2] * flux[0];   // vv, vvfo
        volt[0] = flux[0];
        flux[0] = f;
    }
}

void Engine_Ext_United_UPML::DoPreVoltageUpdates(int threadID)
{
    if (threadID != 0)  return;

    if (m_mgpu) {
        Engine_cuda_mgpu *mg = static_cast<Engine_cuda_mgpu*>(m_Eng);
        for (int g = 0; g < mg->NumSlabs(); ++g) {
            if (mg_nblocks[g] == 0) continue;
            const CudaSlabCtx &c = mg->Slab(g);
            cudaSetDevice(c.device);
            dim3 blk((mg_maxcells[g] * 3 + 1023) / 1024, mg_nblocks[g]);
            PreVoltageUpdateKernel<<<blk, 1024, 0, c.stream>>>(c.d_volt, mg_blks[g], c.local_dim);
        }
        return;
    }

    Engine_cuda *eng = static_cast<Engine_cuda*>(m_Eng);
    int upml_blocks = m_Op_UPML_List->size();
    int tBlockSizeX = (m_max_num_cells_in_block * 3 + 1023) / 1024;

    dim3 blk(tBlockSizeX, upml_blocks);

    PreVoltageUpdateKernel<<<blk, 1024>>>(eng->GetDeviceVoltData(), m_d_blks, eng->GetDeviceDimData());

}


__global__
void PostVoltageUpdateKernel(FDTD_FLOAT *d_volt, const upml_block_t *ublk, const dim3 dim) 
{
    const upml_block_t area = ublk[blockIdx.y];
    int N = area.lines.x * area.lines.y * area.lines.z * 3;

    for (auto i : hemi::grid_stride_range(0, N)) {
        int gi = flat_indx_in_full_grid(i / 3, &area, &dim) * 3 + (i % 3);

        FDTD_FLOAT *volt = d_volt + gi;
        FDTD_FLOAT vvfn = area.coeff_table[(int)area.coeff_index[i] * 6 + 1];   // vvfn
        FDTD_FLOAT *flux = area.volt_flux + i;

        FDTD_FLOAT f = flux[0];
        flux[0] = volt[0];
        volt[0] = f + vvfn * volt[0];
    }
}

// Fused PostVoltage(volt,volt_flux) + PreCurrent(curr,curr_flux): the two swap
// chains touch disjoint arrays, share the same launch geometry, and read the
// SAME coeff_index[i]/gi — fusing loads the index and computes gi once, halves
// the kernel launches on this seam, and doubles per-thread memory-level
// parallelism. Bit-identical per-address operation order vs the separate
// kernels; only legal when nothing between the two hooks touches curr (gated
// below on the extension set being exactly {united UPML, excitation}).
__global__
void FusedPostVoltPreCurrKernel(FDTD_FLOAT *d_volt, FDTD_FLOAT *d_curr,
                                const upml_block_t *ublk, const dim3 dim)
{
    const upml_block_t area = ublk[blockIdx.y];
    int N = area.lines.x * area.lines.y * area.lines.z * 3;

    for (auto i : hemi::grid_stride_range(0, N)) {
        int gi = flat_indx_in_full_grid(i / 3, &area, &dim) * 3 + (i % 3);
        const FDTD_FLOAT *c = area.coeff_table + (int)area.coeff_index[i] * 6;

        // PostVoltage chain (vvfn = c[1])
        FDTD_FLOAT *volt  = d_volt + gi;
        FDTD_FLOAT *vflux = area.volt_flux + i;
        FDTD_FLOAT fv = vflux[0];
        vflux[0] = volt[0];
        volt[0] = fv + c[1] * volt[0];

        // PreCurrent chain (ii = c[3], iifo = c[5])
        FDTD_FLOAT *curr  = d_curr + gi;
        FDTD_FLOAT *cflux = area.curr_flux + i;
        FDTD_FLOAT fc = c[3] * curr[0] - c[5] * cflux[0];
        curr[0] = cflux[0];
        cflux[0] = fc;
    }
}

void Engine_Ext_United_UPML::DoPostVoltageUpdates(int threadID)
{
    if (threadID != 0)  return;

    if (m_mgpu) {
        Engine_cuda_mgpu *mg = static_cast<Engine_cuda_mgpu*>(m_Eng);
        for (int g = 0; g < mg->NumSlabs(); ++g) {
            if (mg_nblocks[g] == 0) continue;
            const CudaSlabCtx &c = mg->Slab(g);
            cudaSetDevice(c.device);
            dim3 blk((mg_maxcells[g] * 3 + 1023) / 1024, mg_nblocks[g]);
            PostVoltageUpdateKernel<<<blk, 1024, 0, c.stream>>>(c.d_volt, mg_blks[g], c.local_dim);
        }
        return;
    }

    Engine_cuda *eng = static_cast<Engine_cuda*>(m_Eng);
    int upml_blocks = m_Op_UPML_List->size();
    int tBlockSizeX = (m_max_num_cells_in_block * 3 + 1023) / 1024;

    dim3 blk (tBlockSizeX, upml_blocks);

    // Decide the fused fast path once (before the graph is captured): fusing
    // hoists the PreCurrent curr/flux swap before Apply2Voltages, which is a
    // pure reorder ONLY if no other extension's hooks touch curr in between.
    // Excitation's Apply2Voltages adds to volt only, so {UPML, excitation} is
    // safe; anything else (Mur/RLC/dispersive/TFSF/steadystate) falls back to
    // the separate, canonical kernel order.
    if (!m_fuse_checked) {
        m_fuse_checked = true;
        m_fuse = true;
        for (size_t n = 0; n < m_Eng->GetExtensionCount(); ++n) {
            Engine_Extension* e = m_Eng->GetExtension(n);
            if (e == this) continue;
            if (dynamic_cast<Engine_Ext_Excitation*>(e)) continue;
            m_fuse = false;
            break;
        }
    }

    if (m_fuse)
        FusedPostVoltPreCurrKernel<<<blk, 1024>>>(eng->GetDeviceVoltData(), eng->GetDeviceCurrData(),
                                                  m_d_blks, eng->GetDeviceDimData());
    else
        PostVoltageUpdateKernel<<<blk, 1024>>>(eng->GetDeviceVoltData(), m_d_blks, eng->GetDeviceDimData());
}

__global__ 
void PreCurrentUpdateKernel(FDTD_FLOAT *d_curr, const upml_block_t *ublk, const dim3 dim) 
{
    // we use blockIdx.y to determine which block we are in, 
    // blockIdx.x and threadIdx.x to get the data index within that block
    const upml_block_t area = ublk[blockIdx.y];
    int N = area.lines.x * area.lines.y * area.lines.z * 3;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // find the cell location in whole matrix
        int gi = flat_indx_in_full_grid(i / 3, &area, &dim) * 3 + (i % 3);

        const FDTD_FLOAT *c = area.coeff_table + (int)area.coeff_index[i] * 6;
        FDTD_FLOAT *curr = d_curr + gi;
        FDTD_FLOAT *flux = area.curr_flux + i;

        FDTD_FLOAT f = c[3] * curr[0] - c[5] * flux[0];   // ii, iifo
        curr[0] = flux[0];
        flux[0] = f;
    }
}


void Engine_Ext_United_UPML::DoPreCurrentUpdates(int threadID)
{
    if (threadID != 0)  return;

    if (m_mgpu) {
        Engine_cuda_mgpu *mg = static_cast<Engine_cuda_mgpu*>(m_Eng);
        for (int g = 0; g < mg->NumSlabs(); ++g) {
            if (mg_nblocks[g] == 0) continue;
            const CudaSlabCtx &c = mg->Slab(g);
            cudaSetDevice(c.device);
            dim3 blk((mg_maxcells[g] * 3 + 1023) / 1024, mg_nblocks[g]);
            PreCurrentUpdateKernel<<<blk, 1024, 0, c.stream>>>(c.d_curr, mg_blks[g], c.local_dim);
        }
        return;
    }

    if (m_fuse) return;   // already done inside FusedPostVoltPreCurrKernel

    Engine_cuda *eng = static_cast<Engine_cuda*>(m_Eng);
    int upml_blocks = m_Op_UPML_List->size();

    int tBlockSizeX = (m_max_num_cells_in_block * 3 + 1023) / 1024;

    dim3 blk (tBlockSizeX, upml_blocks);

    PreCurrentUpdateKernel<<<blk, 1024>>>(eng->GetDeviceCurrData(), m_d_blks, eng->GetDeviceDimData());

}


__global__
void PostCurrentUpdateKernel(FDTD_FLOAT *d_curr, const upml_block_t *ublk, const dim3 dim)
{
    // Component-per-thread (i = cell*3+comp), matching the other 3 PML kernels,
    // so this kernel launches 3x the threads -> more memory-level parallelism to
    // hide the scattered curr[] access latency. Same traffic, logic unchanged.
    const upml_block_t area = ublk[blockIdx.y];
    int N = area.lines.x * area.lines.y * area.lines.z * 3;

    for (auto i : hemi::grid_stride_range(0, N)) {
        int gi = flat_indx_in_full_grid(i / 3, &area, &dim) * 3 + (i % 3);

        FDTD_FLOAT *curr = d_curr + gi;
        FDTD_FLOAT *flux = area.curr_flux + i;
        FDTD_FLOAT iifn = area.coeff_table[(int)area.coeff_index[i] * 6 + 4];   // iifn

        FDTD_FLOAT f = flux[0];
        flux[0] = curr[0];
        curr[0] = f + iifn * flux[0];
    }
}

void Engine_Ext_United_UPML::DoPostCurrentUpdates(int threadID)
{
    if (threadID != 0)  return;

    if (m_mgpu) {
        Engine_cuda_mgpu *mg = static_cast<Engine_cuda_mgpu*>(m_Eng);
        for (int g = 0; g < mg->NumSlabs(); ++g) {
            if (mg_nblocks[g] == 0) continue;
            const CudaSlabCtx &c = mg->Slab(g);
            cudaSetDevice(c.device);
            dim3 blk((mg_maxcells[g] * 3 + 1023) / 1024, mg_nblocks[g]);
            PostCurrentUpdateKernel<<<blk, 1024, 0, c.stream>>>(c.d_curr, mg_blks[g], c.local_dim);
        }
        return;
    }

    Engine_cuda *eng = static_cast<Engine_cuda*>(m_Eng);
    int upml_blocks = m_Op_UPML_List->size();
    int tBlockSizeX = (m_max_num_cells_in_block * 3 + 1023) / 1024;

    dim3 blk (tBlockSizeX, upml_blocks);

    PostCurrentUpdateKernel<<<blk, 1024>>>(eng->GetDeviceCurrData(), m_d_blks, eng->GetDeviceDimData());

}


