#ifndef ENGINE_CUDA_MGPU_H
#define ENGINE_CUDA_MGPU_H

#include "engine_cuda.h"

#include <vector>
#include <unordered_set>

//! One GPU's slab of the simulation domain.
//
// The grid is cut into contiguous slabs along x (the outermost axis of the
// flat (x*ny*nz + y*nz + z)*3+comp layout, so every slab -- and every halo
// plane -- is a contiguous memory range). Each slab's local field arrays hold
// nx_real real planes bracketed by one ghost plane on each side:
//   local plane 0            = ghost  (global plane x_start-1, from left nbr)
//   local planes 1..nx_real  = real   (global planes x_start .. x_end-1)
//   local plane nx_real+1    = ghost  (global plane x_end, from right nbr)
// Ghost planes at the global domain edges exist but are unused (the update
// kernels apply the global boundary rules there instead).
struct CudaSlabCtx
{
	int          device;        // CUDA device ordinal (all equal in virtual mode)
	cudaStream_t stream;        // this slab's work stream
	FDTD_FLOAT  *d_volt;        // local volt, (nx_real+2)*ny*nz*3 floats
	FDTD_FLOAT  *d_curr;        // local curr, same size
	int          x_start;       // first owned global x plane (inclusive)
	int          x_end;         // one past last owned global x plane
	dim3         local_dim;     // (nx_real+2, ny, nz) -- for extension kernels
	int         *d_numTS;       // per-slab device timestep counter
};

//! Multi-GPU CUDA FDTD engine: slab decomposition with per-timestep halo exchange.
//
// Per timestep and slab g (everything issued async on stream[g], no host sync):
//   1. wait curr-halo event from g-1 (its last curr plane -> our left ghost)
//   2. PML PreVolt; voltage update; PML PostVolt; voltage excitation
//   3. send our first volt plane -> g-1's right ghost; record volt event
//      wait volt-halo event from g+1
//   4. PML PreCurr; current update; PML PostCurr; current excitation
//   5. send our last curr plane -> g+1's left ghost; record curr event
// Events are ping-ponged by timestep parity so replays never overtake.
// Halo exchange is numerically exact, so results are bit-identical to the
// single-GPU engine (same per-cell operation order).
//
// Only engaged when the operator's extension set is exactly
// {UPML, excitation}; anything else falls back to the single-GPU engine.
class Engine_cuda_mgpu : public Engine_cuda
{
public:
	//! num_slabs GPUs; if virtual_mode, all slabs share device base_device (for
	//! validating the decomposition/halo logic on a single-GPU machine).
	static Engine_cuda_mgpu* New(const Operator_CUDA* op, int num_slabs,
	                             bool virtual_mode, unsigned int base_device);
	virtual ~Engine_cuda_mgpu();

	virtual void Init();
	virtual void Reset();

	virtual bool IterateTS(unsigned int iterTS);
	virtual double CalcFastEnergy();

	virtual void AddVolt(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value);
	virtual void AddCurr(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value);

	// Slab access for the mgpu-aware extensions (UPML, excitation).
	int NumSlabs() const { return (int)m_ctx.size(); }
	const CudaSlabCtx& Slab(int i) const { return m_ctx[i]; }

	// The single-device accessors have no meaning here; abort loudly so any
	// extension that was not made mgpu-aware cannot silently corrupt physics.
	virtual FDTD_FLOAT* GetDeviceVoltData();
	virtual FDTD_FLOAT* GetDeviceCurrData();
	virtual int* GetDeviceNumTS();

	// Host mirror access (probes/dumps); fetches owning-slab data on demand.
	virtual FDTD_FLOAT GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const;
	virtual FDTD_FLOAT GetVolt(unsigned int n, const unsigned int pos[3]) const;
	virtual FDTD_FLOAT GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const;
	virtual FDTD_FLOAT GetCurr(unsigned int n, const unsigned int pos[3]) const;
	virtual void SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val);
	virtual void SetVolt(unsigned int n, const unsigned int pos[3], FDTD_FLOAT val);
	virtual void SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val);
	virtual void SetCurr(unsigned int n, const unsigned int pos[3], FDTD_FLOAT val);

protected:
	Engine_cuda_mgpu(const Operator_CUDA* op, int num_slabs, bool virtual_mode);

	virtual void UpdateVoltages(unsigned int startX, unsigned int numX);
	virtual void UpdateCurrents(unsigned int startX, unsigned int numX);

private:
	void RunOneTimestepMg(int parity);
	void HaloSendVolt(int parity);   // step 3 sends+records (all slabs)
	void HaloSendCurr(int parity);   // step 5 sends+records (all slabs)
	int  OwnerSlab(int x) const;
	void EnsureCellHostMg(int cell) const;
	void SelectiveReadbackMg();
	void SelectiveReadbackFinishMg();
	void UploadHostMirror();

	int  m_num_slabs;
	bool m_virtual;
	std::vector<CudaSlabCtx> m_ctx;

	// per-slab coefficient data (compressed, sliced from the operator's arrays)
	std::vector<void*>        m_d_idx;
	std::vector<bool>         m_idx_u16;
	std::vector<FDTD_FLOAT*>  m_d_vvvi;
	std::vector<FDTD_FLOAT*>  m_d_iiiv;

	// halo events, ping-ponged by timestep parity: [slab][parity]
	std::vector<cudaEvent_t>  m_ev_volt[2];
	std::vector<cudaEvent_t>  m_ev_curr[2];
	long long                 m_step_counter = 0;

	// per-slab energy accumulators + pinned result
	std::vector<double*>      m_d_energy;
	double                   *m_h_energy = NULL;   // pinned, one per slab

	// full-size pinned host field mirror (probes/dumps read this)
	FDTD_FLOAT *m_h_volt = NULL;
	FDTD_FLOAT *m_h_curr = NULL;
	int  m_volt_dirty = 0;      // host wrote the mirror -> re-upload
	int  m_curr_dirty = 0;
	bool m_locked = false;

	// selective readback (per-slab batched gather, mirrors the single-GPU path)
	mutable std::unordered_set<int> m_cells;
	mutable bool m_full = false;
	mutable bool m_dirty = true;
	bool m_pending = false;
	std::vector<std::vector<int> > m_cell_list;    // per slab, LOCAL flat cell ids
	std::vector<std::vector<int> > m_cell_glob;    // per slab, matching GLOBAL ids
	std::vector<int*>         m_d_cells;
	std::vector<FDTD_FLOAT*>  m_d_stage;
	std::vector<FDTD_FLOAT*>  m_h_stage;
	std::vector<size_t>       m_stage_cap;
};

#endif // ENGINE_CUDA_MGPU_H
