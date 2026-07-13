#ifndef ENGINE_CUDA_H
#define ENGINE_CUDA_H

#include "engine.h"
#include "operator_cuda.h"

#include "tools/cuda/array.h"
#include "Common/field_gather_backend.h"

#include <unordered_set>
#include <vector>
#include <mutex>


class Operator_CUDA;

class Engine_cuda : public Engine, public FieldGatherBackend
{
public:
	static Engine_cuda* New(const Operator_CUDA* op, unsigned int cuda_device_number);
	virtual ~Engine_cuda();

	virtual void Init();
	virtual void Reset();

	virtual void setCUDAdevice(unsigned int cuda_device_number);

	//!Iterate a number of timesteps
	virtual bool IterateTS(unsigned int iterTS);

	//! Pipelined chunks: in full-mirror readback mode the chunk is queued on
	//! the stream and runs while the caller processes the previous chunk's
	//! host state; WaitChunk() then syncs, reads the fields back and advances
	//! numTS. In selective mode (probes may still learn new mirror cells,
	//! which needs an idle device at the presented timestep) the chunk is
	//! deferred and executed synchronously inside WaitChunk() -- identical to
	//! the classic serial loop.
	virtual bool SupportsAsyncChunks() const {return true;}
	virtual void LaunchChunkAsync(unsigned int iterTS);
	virtual bool WaitChunk();

	// --- FieldGatherBackend: on-device field-dump interpolation -----------
	virtual int RegisterFieldGather(const std::vector<unsigned int>& offsets,
	                                const std::vector<unsigned int>& src,
	                                const std::vector<float>& coeff,
	                                bool useCurr, size_t nOut, float** hostOut);
	virtual long GetGatherTS(int id) const;

	int inline FlatIndex(int x, int y, int z) const {
		return x * numLines[1] * numLines[2] + y * numLines[2] + z; 
	}


	unsigned int m_cuda_device_number;
	int m_supports_coop_launch;

	// env OPENEMS_PROF=1: cumulative seconds spent in the chunk-end readback
	// (D2H copies + device sync), readable by the run loop for reporting.
	double m_prof_readback = 0.0;

	virtual double CalcFastEnergy();

	virtual void AddVolt(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value);
	virtual void AddCurr(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value);

	virtual inline FDTD_FLOAT* GetDeviceVoltData() { return volt_array->device_data(); }
	virtual inline FDTD_FLOAT* GetDeviceCurrData() { return curr_array->device_data(); }
	virtual inline dim3 GetDeviceDimData() { return m_dim; }
	// Device-side timestep counter driving the excitation kernels; lets the
	// per-timestep sequence be captured once into a CUDA graph and replayed.
	virtual inline int* GetDeviceNumTS() { return d_numTS; }

	//this access functions muss be overloaded by any new engine using a different storage model
#if 1
	inline virtual FDTD_FLOAT GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		assert(!m_host_data_locked);
		EnsureCellHost(FlatIndex(x, y, z));
		FDTD_FLOAT *volt = volt_array->host_data();
		return volt[FlatIndex(x, y, z) * 3 + n];
	}

	inline virtual FDTD_FLOAT GetVolt(unsigned int n, const unsigned int pos[3]) const
	{
		return GetVolt(n, pos[0], pos[1], pos[2]);
	}

	inline virtual FDTD_FLOAT GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		assert(!m_host_data_locked);
		EnsureCellHost(FlatIndex(x, y, z));
		return curr_array->host_data()[FlatIndex(x, y, z) * 3 + n];
	}

	inline virtual FDTD_FLOAT GetCurr(unsigned int n, const unsigned int pos[3]) const
	{
		return GetCurr(n, pos[0], pos[1], pos[2]);
	}

	inline virtual void SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val)
	{
		assert(!m_host_data_locked);
		volt_array->host_data()[FlatIndex(x, y, z) * 3 + n] = val;
		m_volt_updated += 1;
	}

	inline virtual void SetVolt(unsigned int n, const unsigned int pos[3], FDTD_FLOAT val)
	{
		SetVolt(n, pos[0], pos[1], pos[2], val);
	}

	inline virtual void SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val)
	{
		assert(!m_host_data_locked);
		curr_array->host_data()[FlatIndex(x, y, z) * 3 + n] = val;
		m_curr_updated += 1;
	}

	inline virtual void SetCurr(unsigned int n, const unsigned int pos[3], FDTD_FLOAT val)
	{
		SetCurr(n, pos[0], pos[1], pos[2], val);
	}
#endif


protected:

	Engine_cuda(const Operator_CUDA* op);
	const Operator_CUDA* Op;

	virtual void UpdateVoltages(unsigned int startX, unsigned int numX);
	virtual void UpdateCurrents(unsigned int startX, unsigned int numX);

	inline unsigned int getLinearIndex(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		return (x * (numLines[1] * numLines[2]) + y * numLines[2] + z) * 3 + n;
	}

	inline unsigned int getLinearIndex(unsigned int n, const unsigned int pos[3]) const
	{
		return getLinearIndex(n, pos[0], pos[1], pos[2]);
	}

	// IterateTS split into its async-queueable front half (upload + graph
	// launches, no sync) and its finishing half (readback + sync + numTS).
	void LaunchChunkBody(unsigned int iterTS);
	void FinishChunkBody(unsigned int iterTS);
	unsigned int m_async_pending = 0;   // timesteps queued by LaunchChunkAsync()

	// On-device field-dump gathers (registered by ProcessFields). Each holds a
	// CSR interpolation stencil; RunFieldGathers() evaluates them from the
	// device volt/curr arrays at chunk finish into pinned host buffers.
	struct GpuGather {
		unsigned int *d_offsets;  // nOut+1
		unsigned int *d_src;      // nEntries
		float        *d_coeff;    // nEntries
		float        *d_out;      // nOut (device)
		float        *h_out;      // nOut (pinned host, handed to ProcessFields)
		size_t        nOut;
		bool          useCurr;
		long          ts;         // numTS the h_out buffer currently holds
	};
	std::vector<GpuGather> m_gathers;
	void RunFieldGathers();      // enqueue all gathers on the work stream
	void FreeFieldGathers();

	double ComputeDeviceEnergy();          // raw device energy reduction (syncs)
	double m_energy_cache = 0.0;           // energy at last chunk finish
	bool   m_energy_valid = false;         // cache holds a chunk-consistent value

	dim3 m_dim;

	FDTD_FLOAT *d_fastEnergy;

	// Compressed coefficients: a per-cell index into small unique-coefficient
	// tables (see Engine_cuda::Init). Replaces the full per-cell vv/vi/ii/iv
	// arrays; the tables are tiny (fit in L2) so the update kernels move a 4-byte
	// index instead of 24 bytes of coefficients per cell per field.
	void         *d_op_index;   // per-cell index; uint16 (m_index_u16) or uint32
	bool          m_index_u16;
	FDTD_FLOAT   *d_vv_vi_table;
	FDTD_FLOAT   *d_ii_iv_table;
	unsigned int  m_num_unique_coeff;

private:

	CudaHelper::Array<FDTD_FLOAT> *volt_array;
	CudaHelper::Array<FDTD_FLOAT> *curr_array;
	CudaHelper::Array<double> *m_energy_sum;

	int m_volt_updated;
	uint32_t m_volt_updated_by_host;

	int m_curr_updated;
	uint32_t m_curr_updated_by_host;

	bool m_host_data_locked;

	// --- CUDA graph capture of the per-timestep kernel sequence ---
	int          *d_numTS;        // device timestep counter (read by excitation kernels)
	cudaGraph_t   m_graph;
	cudaGraphExec_t m_graphExec;
	bool          m_graph_ready;

	void RunOneTimestep();        // one timestep's kernel launches (no readback)

	// --- Selective field readback ---
	// The host field mirror is only needed at the handful of cells that probes
	// and dumps read through GetVolt/GetCurr. Instead of copying the whole
	// volt/curr arrays back every chunk, learn that cell footprint on demand and
	// refresh only those cells. Falls back to a full readback if the footprint
	// grows large (e.g. a field dump over a big region).
	mutable std::mutex m_rb_mtx;                  // guards cell learning (parallel dump extraction)
	mutable std::unordered_set<int> m_rb_cells;   // cell indices the mirror must track
	mutable bool m_rb_fullmode;                    // true -> copy the whole array
	void EnsureCellHost(int cell) const;           // on-demand fetch of a first-seen cell
	void SelectiveReadback();                       // issue batched gather + one D2H copy
	void SelectiveReadbackFinish();                 // scatter staged data into the mirror

	// Batched-readback staging: a gather kernel packs volt+curr of every tracked
	// cell into d_rb_stage, one D2H copy lands it in pinned h_rb_stage, and after
	// the chunk-end sync SelectiveReadbackFinish scatters into the host mirror.
	// (Replaces 2 tiny cudaMemcpyAsync per cell per chunk.)
	mutable bool  m_rb_dirty = true;               // cell set changed -> re-upload list
	bool          m_rb_pending = false;            // a staged copy awaits scatter
	int           m_rb_count = 0;
	size_t        m_rb_capacity = 0;
	std::vector<int> m_rb_list;                    // host copy of the cell list
	int          *d_rb_list = NULL;
	FDTD_FLOAT   *d_rb_stage = NULL;               // 6 floats per cell (volt3 + curr3)
	FDTD_FLOAT   *h_rb_stage = NULL;               // pinned

	void checkZero();

};

#endif // ENGINE_CUDA_H
