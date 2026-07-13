/*
*	CUDA port of the lumped-RLC ADE extension. Mirrors the host recurrence in
*	engine_ext_lumpedRLC.cpp exactly, but as on-device kernels so it is captured
*	into the per-timestep CUDA graph. The host's pointer-rotated Vd/J history is
*	linearised into explicit device slots (Vd1=Vd[n-1], Vd2=Vd[n-2], etc.).
*	No host-side field access.
*/

#include "engine_ext_lumpedRLC.h"
#include "operator_ext_lumpedRLC.h"
#include "FDTD/engine_cuda.h"

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
// from the RLC recurrence, write it back, then shift the history slots.
__global__
void rlcApplyKernel(FDTD_FLOAT* volt, dim3 dim, const rlc_cell* c,
	const FDTD_FLOAT* Il, FDTD_FLOAT* Vd1, FDTD_FLOAT* Vd2,
	FDTD_FLOAT* J1, FDTD_FLOAT* J2, int N)
{
	for (auto k : hemi::grid_stride_range(0, N)) {
		rlc_cell e = c[k];
		int flat = (e.x * dim.y * dim.z + e.y * dim.z + e.z) * 3 + e.dir;

		FDTD_FLOAT il = Il[k], vd1 = Vd1[k], vd2 = Vd2[k], j1 = J1[k], j2 = J2[k];
		FDTD_FLOAT Veng = volt[flat];

		// exact host associativity: ((((Veng - Il) + vv2*Vd2) + vj1*J1) + vj2*J2)
		FDTD_FLOAT Vd_n = e.vvd * (Veng - il + e.vv2 * vd2 + e.vj1 * j1 + e.vj2 * j2);
		FDTD_FLOAT J_n  = e.ib0 * (Vd_n - vd2) - (e.b1 * e.ib0) * j1 - (e.b2 * e.ib0) * j2;

		volt[flat] = Vd_n;

		// shift history for the next timestep (reads done above)
		Vd2[k] = vd1;  Vd1[k] = Vd_n;
		J2[k]  = j1;   J1[k]  = J_n;
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
		h[i].vv2 = op->v_RLC_vv2[i];
		h[i].vj1 = op->v_RLC_vj1[i];
		h[i].vj2 = op->v_RLC_vj2[i];
		h[i].vvd = op->v_RLC_vvd[i];
		h[i].ib0 = op->v_RLC_ib0[i];
		h[i].b1  = op->v_RLC_b1[i];
		h[i].b2  = op->v_RLC_b2[i];
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
	int N = (int)m_Op_Ext_RLC->RLC_count;
	int blocks = (N + RLC_THREADS - 1) / RLC_THREADS;
	rlcPreKernel<<<blocks, RLC_THREADS>>>(d_cells, d_Il, d_Vd1, N);
}

void Engine_Ext_LumpedRLC::Apply2VoltagesCuda(Engine_cuda* eng)
{
	int N = (int)m_Op_Ext_RLC->RLC_count;
	int blocks = (N + RLC_THREADS - 1) / RLC_THREADS;
	rlcApplyKernel<<<blocks, RLC_THREADS>>>(
		eng->GetDeviceVoltData(), eng->GetDeviceDimData(), d_cells,
		d_Il, d_Vd1, d_Vd2, d_J1, d_J2, N);
}
