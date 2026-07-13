/*
*	Copyright (C) 2025 Tommy Gu (radiotommy@gmail.com)
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ENGINE_EXT_UNITED_UPML_H
#define ENGINE_EXT_UNITED_UPML_H

#include "engine_extension.h"
#include "FDTD/engine.h"
#include "FDTD/operator.h"
#include "engine_extension_dispatcher.h"

#include "tools/cuda/array.h"

#include <vector>
#include <utility>

struct upml_block_t {
	dim3 start;
	dim3 lines;
    FDTD_FLOAT *volt_flux;
    FDTD_FLOAT *curr_flux;
    // Compressed coefficients: the 6 per-(cell,component) PML coeffs vary only
    // with depth into the absorber, so they dedup strongly. Each (cell,comp)
    // stores one uint16 index into a shared L2-resident table of unique
    // sextets laid out [vv, vvfo, vvfn, ii, iifo, iifn] (grouped so each kernel
    // reads a contiguous 1- or 2-float span).
    const unsigned short *coeff_index;   // per (cell*3+comp)
    const FDTD_FLOAT     *coeff_table;   // shared across all blocks; 6 floats/entry
};


class Operator_Ext_UPML;

class Engine_Ext_United_UPML : public Engine_Extension
{
public:
    Engine_Ext_United_UPML(std::vector<Operator_Ext_UPML *> *op_ext_upml_list);
	virtual ~Engine_Ext_United_UPML();

	virtual void SetEngine(Engine* eng);
#if WITH_CUDA
	virtual bool IsCUDACapable() const {return true;}
#endif

	virtual void DoPreVoltageUpdates() {Engine_Ext_United_UPML::DoPreVoltageUpdates(0);};
	virtual void DoPreVoltageUpdates(int threadID);
	virtual void DoPostVoltageUpdates() {Engine_Ext_United_UPML::DoPostVoltageUpdates(0);};
	virtual void DoPostVoltageUpdates(int threadID);

	virtual void DoPreCurrentUpdates() {Engine_Ext_United_UPML::DoPreCurrentUpdates(0);};
	virtual void DoPreCurrentUpdates(int threadID);
	virtual void DoPostCurrentUpdates() {Engine_Ext_United_UPML::DoPostCurrentUpdates(0);};
	virtual void DoPostCurrentUpdates(int threadID);


protected:
    template <typename EngType>
    void DoPreVoltageUpdatesImpl(EngType* eng, int threadID);

    template <typename EngType>
    void DoPostVoltageUpdatesImpl(EngType* eng, int threadID);

    template <typename EngType>
    void DoPreCurrentUpdatesImpl(EngType* eng, int threadID);

    template <typename EngType>
    void DoPostCurrentUpdatesImpl(EngType* eng, int threadID);

private:
	std::vector<Operator_Ext_UPML *> *m_Op_UPML_List;

    // location and size of each UPML block
    upml_block_t *m_d_blks;
    // pre calculate how many cells are in each block
    int *m_num_of_cells_in_block;
    int m_max_num_cells_in_block;

    std::vector<CudaHelper::Array<FDTD_FLOAT> *> m_volt_fluxes;
    std::vector<CudaHelper::Array<FDTD_FLOAT> *> m_curr_fluxes;

    // Compressed PML coefficients: one shared unique-sextet table + one uint16
    // index array per block (freed in the destructor).
    FDTD_FLOAT *m_d_coeff_table = NULL;
    std::vector<unsigned short *> m_d_coeff_index;

    // Fused PostVoltage+PreCurrent fast path (see .cu): only enabled when the
    // engine's extension set is exactly {this united UPML, excitation}, so
    // hoisting the curr/flux swap before Apply2Voltages is a pure reorder of
    // operations on disjoint addresses (bit-identical results).
    bool m_fuse = false;
    bool m_fuse_checked = false;

    // multi-GPU: PML blocks clipped to each slab's x-range and uploaded per
    // device (block starts translated to slab-local coordinates, flux state
    // owned per slab, coefficient table deduped per slab). Empty when
    // running single-GPU.
    bool m_mgpu = false;
    std::vector<upml_block_t*> mg_blks;       // per slab: device block array
    std::vector<int>           mg_nblocks;    // per slab: clipped block count
    std::vector<int>           mg_maxcells;   // per slab: max cells over its blocks
    std::vector<std::pair<int,void*> > mg_allocs;   // (device, ptr) for cleanup
    void SetEngineMg(class Engine_cuda_mgpu* mg);

};

#endif