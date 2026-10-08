/*
*	CUDA port of the lumped-RLC ADE extension. Mirrors the host update in
*	engine_ext_lumpedRLC.cpp, but as on-device kernels so it is captured into
*	the per-timestep CUDA graph. The host's pointer-rotated Vd/J history is
*	linearised into explicit device slots (Vd1=Vd[n-1], J1=J[n-1]; the Vd2
*	slot holds the series charge q of the state-space form).
*	No host-side field access.
*/

#include "engine_ext_lumpedRLC.h"
#include "operator_ext_lumpedRLC.h"
#include "FDTD/engine_cuda.h"
#include "FDTD/engine_cuda_mgpu.h"

#include <cuda_runtime.h>
#include "hemi/grid_stride_range.h"
#include "tools/cuda/check.h"

#define RLC_THREADS 128

typedef Engine_Ext_LumpedRLC::rlc_cell rlc_cell;

// Pre (before core update): accumulate parallel-inductor current using Vd[n-1].
__global__
void rlcPreKernel(const rlc_cell* c, FDTD_FLOAT* Il, const FDTD_FLOAT* Vd1, int N)
{
	for (auto k : hemi::grid_stride_range(0, N)) {
		Il[k] = Il[k] + c[k].i2v * c[k].ilv * Vd1[k];
	}
}

// Apply (after core update): read the engine node voltage, form Vd[n] and J[n]
// with the trapezoidal state-space update (same as the host), write it back,
// then advance the state. Slots: Vd1 = Vd[n-1], J1 = J[n-1], Q = series charge.
__global__
void rlcApplyKernel(FDTD_FLOAT* volt, dim3 dim, const rlc_cell* c,
	const FDTD_FLOAT* Il, FDTD_FLOAT* Vd1, FDTD_FLOAT* Q,
	FDTD_FLOAT* J1, FDTD_FLOAT dT_half, int N)
{
	for (auto k : hemi::grid_stride_range(0, N)) {
		rlc_cell e = c[k];
		int flat = (e.x * dim.y * dim.z + e.y * dim.z + e.z) * 3 + e.dir;

		FDTD_FLOAT il = Il[k], vd1 = Vd1[k], q = Q[k], j1 = J1[k];
		FDTD_FLOAT Veng = volt[flat];

		FDTD_FLOAT B    = e.aV * vd1 + e.aQ * q + e.aJ * j1;
		FDTD_FLOAT Vd_n = e.vvd * (Veng - il - e.vcd * (B + j1));
		FDTD_FLOAT J_n  = e.dJdV * Vd_n + B;

		volt[flat] = Vd_n;

		Vd1[k] = Vd_n;
		J1[k]  = J_n;
		Q[k]   = q + dT_half * (J_n + j1);
	}
}

// ---- multi-GPU: partition elements by owning slab; per-slab aux state ----
void Engine_Ext_LumpedRLC::SetEngineMg(Engine_cuda_mgpu* mg)
{
	Operator_Ext_LumpedRLC* op = m_Op_Ext_RLC;
	int nslab = mg->NumSlabs();
	mg_cells.assign(nslab, NULL);
	mg_count.assign(nslab, 0);
	mg_dev.assign(nslab, 0);
	mg_Il.assign(nslab, NULL);  mg_Vd1.assign(nslab, NULL); mg_Vd2.assign(nslab, NULL);
	mg_J1.assign(nslab, NULL);  mg_J2.assign(nslab, NULL);

	for (int g = 0; g < nslab; ++g)
	{
		const CudaSlabCtx &c = mg->Slab(g);
		mg_dev[g] = c.device;
		std::vector<rlc_cell> cells;
		for (unsigned int i = 0; i < op->RLC_count; ++i)
		{
			int x = (int)op->v_RLC_pos[0][i];
			if (x < c.x_start || x >= c.x_end) continue;
			rlc_cell h;
			h.x   = x - c.x_start + 1;   // slab-local incl. ghost offset
			h.y   = (int)op->v_RLC_pos[1][i];
			h.z   = (int)op->v_RLC_pos[2][i];
			h.dir = op->v_RLC_dir[i];
			h.ilv = op->v_RLC_ilv[i];  h.i2v = op->v_RLC_i2v[i];
			h.dJdV = op->v_RLC_dJdV[i]; h.aV  = op->v_RLC_aV[i];
			h.aQ   = op->v_RLC_aQ[i];   h.aJ  = op->v_RLC_aJ[i];
			h.vcd  = op->v_RLC_vcd[i];  h.vvd = op->v_RLC_vvd[i];
			cells.push_back(h);
		}
		mg_count[g] = (int)cells.size();
		if (cells.empty()) continue;

		checkCuda(cudaSetDevice(c.device));
		size_t abytes = cells.size() * sizeof(FDTD_FLOAT);
		checkCuda(cudaMalloc(&mg_cells[g], cells.size() * sizeof(rlc_cell)));
		checkCuda(cudaMemcpy(mg_cells[g], cells.data(), cells.size() * sizeof(rlc_cell), cudaMemcpyHostToDevice));
		checkCuda(cudaMalloc(&mg_Il[g],  abytes)); checkCuda(cudaMemset(mg_Il[g],  0, abytes));
		checkCuda(cudaMalloc(&mg_Vd1[g], abytes)); checkCuda(cudaMemset(mg_Vd1[g], 0, abytes));
		checkCuda(cudaMalloc(&mg_Vd2[g], abytes)); checkCuda(cudaMemset(mg_Vd2[g], 0, abytes));
		checkCuda(cudaMalloc(&mg_J1[g],  abytes)); checkCuda(cudaMemset(mg_J1[g],  0, abytes));
		checkCuda(cudaMalloc(&mg_J2[g],  abytes)); checkCuda(cudaMemset(mg_J2[g],  0, abytes));
	}
}

void Engine_Ext_LumpedRLC::DoPreVoltageUpdatesMg(Engine_cuda_mgpu* mg)
{
	for (int g = 0; g < mg->NumSlabs(); ++g)
	{
		if (mg_count[g] == 0) continue;
		const CudaSlabCtx &c = mg->Slab(g);
		cudaSetDevice(c.device);
		int blocks = (mg_count[g] + RLC_THREADS - 1) / RLC_THREADS;
		rlcPreKernel<<<blocks, RLC_THREADS, 0, c.stream>>>(mg_cells[g], mg_Il[g], mg_Vd1[g], mg_count[g]);
	}
}

void Engine_Ext_LumpedRLC::Apply2VoltagesMg(Engine_cuda_mgpu* mg)
{
	for (int g = 0; g < mg->NumSlabs(); ++g)
	{
		if (mg_count[g] == 0) continue;
		const CudaSlabCtx &c = mg->Slab(g);
		cudaSetDevice(c.device);
		int blocks = (mg_count[g] + RLC_THREADS - 1) / RLC_THREADS;
		rlcApplyKernel<<<blocks, RLC_THREADS, 0, c.stream>>>(
			c.d_volt, c.local_dim, mg_cells[g],
			mg_Il[g], mg_Vd1[g], mg_Vd2[g], mg_J1[g], m_Op_Ext_RLC->m_dT_half, mg_count[g]);
	}
}

void Engine_Ext_LumpedRLC::SetEngine(Engine* eng)
{
	m_Eng = eng;
	if (eng->GetType() != Engine::CUDA)
		return;

	unsigned int N = m_Op_Ext_RLC->RLC_count;
	if (!N)
		return;

	if (Engine_cuda_mgpu* mg = dynamic_cast<Engine_cuda_mgpu*>(eng)) {
		SetEngineMg(mg);
		return;
	}

	// pack geometry + coefficients AoS and upload once
	Operator_Ext_LumpedRLC* op = m_Op_Ext_RLC;
	rlc_cell* h = new rlc_cell[N];
	for (unsigned int i = 0; i < N; ++i) {
		h[i].x   = (int)op->v_RLC_pos[0][i];
		h[i].y   = (int)op->v_RLC_pos[1][i];
		h[i].z   = (int)op->v_RLC_pos[2][i];
		h[i].dir = op->v_RLC_dir[i];
		h[i].ilv = op->v_RLC_ilv[i];
		h[i].i2v = op->v_RLC_i2v[i];
		h[i].dJdV = op->v_RLC_dJdV[i];
		h[i].aV   = op->v_RLC_aV[i];
		h[i].aQ   = op->v_RLC_aQ[i];
		h[i].aJ   = op->v_RLC_aJ[i];
		h[i].vcd  = op->v_RLC_vcd[i];
		h[i].vvd  = op->v_RLC_vvd[i];
	}
	checkCuda(cudaMalloc(&d_cells, N * sizeof(rlc_cell)));
	checkCuda(cudaMemcpy(d_cells, h, N * sizeof(rlc_cell), cudaMemcpyHostToDevice));
	delete[] h;

	// aux history, zero-initialised to match the host ctor
	checkCuda(cudaMalloc(&d_Il,  N * sizeof(FDTD_FLOAT)));
	checkCuda(cudaMalloc(&d_Vd1, N * sizeof(FDTD_FLOAT)));
	checkCuda(cudaMalloc(&d_Vd2, N * sizeof(FDTD_FLOAT)));
	checkCuda(cudaMalloc(&d_J1,  N * sizeof(FDTD_FLOAT)));
	checkCuda(cudaMalloc(&d_J2,  N * sizeof(FDTD_FLOAT)));
	checkCuda(cudaMemset(d_Il,  0, N * sizeof(FDTD_FLOAT)));
	checkCuda(cudaMemset(d_Vd1, 0, N * sizeof(FDTD_FLOAT)));
	checkCuda(cudaMemset(d_Vd2, 0, N * sizeof(FDTD_FLOAT)));
	checkCuda(cudaMemset(d_J1,  0, N * sizeof(FDTD_FLOAT)));
	checkCuda(cudaMemset(d_J2,  0, N * sizeof(FDTD_FLOAT)));
}

void Engine_Ext_LumpedRLC::DoPreVoltageUpdatesCuda(Engine_cuda* eng)
{
	if (Engine_cuda_mgpu* mg = dynamic_cast<Engine_cuda_mgpu*>(eng)) {
		DoPreVoltageUpdatesMg(mg);
		return;
	}
	int N = (int)m_Op_Ext_RLC->RLC_count;
	int blocks = (N + RLC_THREADS - 1) / RLC_THREADS;
	rlcPreKernel<<<blocks, RLC_THREADS>>>(d_cells, d_Il, d_Vd1, N);
}

void Engine_Ext_LumpedRLC::Apply2VoltagesCuda(Engine_cuda* eng)
{
	if (Engine_cuda_mgpu* mg = dynamic_cast<Engine_cuda_mgpu*>(eng)) {
		Apply2VoltagesMg(mg);
		return;
	}
	int N = (int)m_Op_Ext_RLC->RLC_count;
	int blocks = (N + RLC_THREADS - 1) / RLC_THREADS;
	rlcApplyKernel<<<blocks, RLC_THREADS>>>(
		eng->GetDeviceVoltData(), eng->GetDeviceDimData(), d_cells,
		d_Il, d_Vd1, d_Vd2, d_J1, m_Op_Ext_RLC->m_dT_half, N);
}
