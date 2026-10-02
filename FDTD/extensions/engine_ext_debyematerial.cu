/*
*	CUDA port of the n-term Debye dispersive-material extension
*	(engine_ext_debyematerial.cpp). The pole update straddles the main voltage
*	update, exactly as on the host:
*
*	  pre   (V = V^n)     h_k = c1_k W_k + d_k V^n;  pre = sum_k (h_k - W_k);  W_k = h_k
*	  main  (V -> V_raw)
*	  post  (V = V_raw)   V^(n+1) = s (V_raw - pre);  ade = V_raw - V^(n+1);  W_k += d_k V^(n+1)
*	  apply               V -= ade
*
*	One thread per active cell does all three components and all poles of that
*	cell, so every array element has a single writer: no atomics, and the
*	arithmetic and its order match the host loops. Coefficients and states are
*	flat device arrays indexed [pole][cell*3 + component]. All on-device, so the
*	sequence is captured into the per-timestep CUDA graph with no host access.
*/

#include "engine_ext_debyematerial.h"
#include "operator_ext_debyematerial.h"
#include "FDTD/engine_cuda.h"
#include "FDTD/engine_cuda_mgpu.h"

#include <climits>
#include <cuda_runtime.h>
#include "hemi/grid_stride_range.h"
#include "tools/cuda/check.h"

#define DEB_THREADS 128

typedef Engine_Ext_DebyeMaterial::deb_cell deb_cell;

__global__
void debyePreVoltKernel(const FDTD_FLOAT* volt, dim3 dim, const deb_cell* c,
	const FDTD_FLOAT* relax, const FDTD_FLOAT* drive, FDTD_FLOAT* W,
	FDTD_FLOAT* pre, int N, int P)
{
	const size_t stride = (size_t)N * 3;
	for (auto k : hemi::grid_stride_range(0, N)) {
		deb_cell e = c[k];
		int base = (e.x * dim.y * dim.z + e.y * dim.z + e.z) * 3;
		#pragma unroll
		for (int n = 0; n < 3; ++n) {
			FDTD_FLOAT v = volt[base + n];
			FDTD_FLOAT acc = 0.0;
			for (int o = 0; o < P; ++o) {
				size_t j = (size_t)o * stride + (size_t)k * 3 + n;
				FDTD_FLOAT w0 = W[j];
				FDTD_FLOAT h = relax[j] * w0 + drive[j] * v;
				acc += h - w0;
				W[j] = h;
			}
			pre[k * 3 + n] = acc;
		}
	}
}

__global__
void debyePostVoltKernel(const FDTD_FLOAT* volt, dim3 dim, const deb_cell* c,
	const FDTD_FLOAT* drive, FDTD_FLOAT* W, const FDTD_FLOAT* pre,
	FDTD_FLOAT* ade, int N, int P)
{
	const size_t stride = (size_t)N * 3;
	for (auto k : hemi::grid_stride_range(0, N)) {
		deb_cell e = c[k];
		int base = (e.x * dim.y * dim.z + e.y * dim.z + e.z) * 3;
		#pragma unroll
		for (int n = 0; n < 3; ++n) {
			FDTD_FLOAT v_raw = volt[base + n];
			FDTD_FLOAT v_new = (v_raw - pre[k * 3 + n]) * e.solve[n];
			ade[k * 3 + n] = v_raw - v_new;
			for (int o = 0; o < P; ++o) {
				size_t j = (size_t)o * stride + (size_t)k * 3 + n;
				W[j] += drive[j] * v_new;
			}
		}
	}
}

// Matches Engine_Ext_Dispersive::Apply2VoltagesImpl. A component with no pole
// has ade exactly 0 and is left untouched (no read-modify-write), as in the
// Lorentz port's apply kernel.
__global__
void debyeApplyKernel(FDTD_FLOAT* volt, dim3 dim, const deb_cell* c,
	const FDTD_FLOAT* ade, int N)
{
	for (auto k : hemi::grid_stride_range(0, N)) {
		deb_cell e = c[k];
		int base = (e.x * dim.y * dim.z + e.y * dim.z + e.z) * 3;
		FDTD_FLOAT a0 = ade[k * 3 + 0];
		FDTD_FLOAT a1 = ade[k * 3 + 1];
		FDTD_FLOAT a2 = ade[k * 3 + 2];
		if (a0 != 0) volt[base + 0] -= a0;
		if (a1 != 0) volt[base + 1] -= a1;
		if (a2 != 0) volt[base + 2] -= a2;
	}
}

// Pack the operator's arrays for the cells with x in [x_lo, x_hi), x shifted by
// x_shift (0 on one GPU; slab-local incl. the ghost offset on multi-GPU). A
// member, because the operator's arrays are visible to this class only.
int Engine_Ext_DebyeMaterial::PackCells(int x_lo, int x_hi, int x_shift,
	std::vector<deb_cell>& cells, std::vector<FDTD_FLOAT>& relax,
	std::vector<FDTD_FLOAT>& drive)
{
	Operator_Ext_DebyeMaterial* op = m_Op_Ext_Deb;
	const int P = op->m_PoleCount;
	unsigned int count = op->m_LM_Count.at(0);
	unsigned int** pos = op->m_LM_pos[0];
	std::vector<unsigned int> idx;
	for (unsigned int i = 0; i < count; ++i) {
		int x = (int)pos[0][i];
		if (x < x_lo || x >= x_hi) continue;
		deb_cell h;
		h.x = x + x_shift;
		h.y = (int)pos[1][i];
		h.z = (int)pos[2][i];
		for (int n = 0; n < 3; ++n)
			h.solve[n] = op->v_solve_ADE[n][i];
		cells.push_back(h);
		idx.push_back(i);
	}
	int N = (int)cells.size();
	relax.assign((size_t)P * N * 3, 0);
	drive.assign((size_t)P * N * 3, 0);
	for (int o = 0; o < P; ++o)
		for (int k = 0; k < N; ++k)
			for (int n = 0; n < 3; ++n) {
				size_t j = (size_t)o * N * 3 + (size_t)k * 3 + n;
				relax[j] = op->v_relax_ADE[o][n][idx[k]];
				drive[j] = op->v_drive_ADE[o][n][idx[k]];
			}
	return N;
}

static void upload(int N, int P, const std::vector<deb_cell>& cells,
	const std::vector<FDTD_FLOAT>& relax, const std::vector<FDTD_FLOAT>& drive,
	deb_cell** d_cells, FDTD_FLOAT** d_relax, FDTD_FLOAT** d_drive,
	FDTD_FLOAT** d_W, FDTD_FLOAT** d_pre, FDTD_FLOAT** d_ade)
{
	size_t poleBytes = (size_t)P * N * 3 * sizeof(FDTD_FLOAT);
	size_t cellBytes = (size_t)N * 3 * sizeof(FDTD_FLOAT);
	checkCuda(cudaMalloc(d_cells, N * sizeof(deb_cell)));
	checkCuda(cudaMemcpy(*d_cells, cells.data(), N * sizeof(deb_cell), cudaMemcpyHostToDevice));
	checkCuda(cudaMalloc(d_relax, poleBytes));
	checkCuda(cudaMemcpy(*d_relax, relax.data(), poleBytes, cudaMemcpyHostToDevice));
	checkCuda(cudaMalloc(d_drive, poleBytes));
	checkCuda(cudaMemcpy(*d_drive, drive.data(), poleBytes, cudaMemcpyHostToDevice));
	// pole states, the pre sum and the correction start at zero, as on the host
	checkCuda(cudaMalloc(d_W, poleBytes));   checkCuda(cudaMemset(*d_W, 0, poleBytes));
	checkCuda(cudaMalloc(d_pre, cellBytes)); checkCuda(cudaMemset(*d_pre, 0, cellBytes));
	checkCuda(cudaMalloc(d_ade, cellBytes)); checkCuda(cudaMemset(*d_ade, 0, cellBytes));
}

void Engine_Ext_DebyeMaterial::SetEngine(Engine* eng)
{
	m_Eng = eng;
	if (eng->GetType() != Engine::CUDA)
		return;
	FreeCuda();
	m_cuda_P = m_Op_Ext_Deb->m_PoleCount;

	if (Engine_cuda_mgpu* mg = dynamic_cast<Engine_cuda_mgpu*>(eng)) {
		SetEngineMg(mg);
		return;
	}

	std::vector<deb_cell> cells;
	std::vector<FDTD_FLOAT> relax, drive;
	m_cuda_N = PackCells(INT_MIN, INT_MAX, 0, cells, relax, drive);
	if (m_cuda_N == 0)
		return;
	upload(m_cuda_N, m_cuda_P, cells, relax, drive,
		&d_cells, &d_relax, &d_drive, &d_W, &d_pre, &d_ade);
}

void Engine_Ext_DebyeMaterial::SetEngineMg(Engine_cuda_mgpu* mg)
{
	int nslab = mg->NumSlabs();
	mg_dev.assign(nslab, 0);
	mg_N.assign(nslab, 0);
	mg_cells.assign(nslab, (deb_cell*)NULL);
	mg_relax.assign(nslab, (FDTD_FLOAT*)NULL);
	mg_drive.assign(nslab, (FDTD_FLOAT*)NULL);
	mg_W.assign(nslab, (FDTD_FLOAT*)NULL);
	mg_pre.assign(nslab, (FDTD_FLOAT*)NULL);
	mg_ade.assign(nslab, (FDTD_FLOAT*)NULL);

	for (int g = 0; g < nslab; ++g)
	{
		const CudaSlabCtx &c = mg->Slab(g);
		mg_dev[g] = c.device;
		checkCuda(cudaSetDevice(c.device));
		std::vector<deb_cell> cells;
		std::vector<FDTD_FLOAT> relax, drive;
		// slab-local x incl. the ghost offset, as in the Lorentz port
		mg_N[g] = PackCells(c.x_start, c.x_end, 1 - c.x_start, cells, relax, drive);
		if (mg_N[g] == 0)
			continue;
		upload(mg_N[g], m_cuda_P, cells, relax, drive,
			&mg_cells[g], &mg_relax[g], &mg_drive[g], &mg_W[g], &mg_pre[g], &mg_ade[g]);
	}
}

void Engine_Ext_DebyeMaterial::DoPreVoltageUpdatesCuda(Engine_cuda* eng)
{
	if (Engine_cuda_mgpu* mg = dynamic_cast<Engine_cuda_mgpu*>(eng)) {
		for (int g = 0; g < mg->NumSlabs(); ++g) {
			if ((int)mg_N.size() <= g || mg_N[g] == 0) continue;
			const CudaSlabCtx &c = mg->Slab(g);
			cudaSetDevice(c.device);
			int blocks = (mg_N[g] + DEB_THREADS - 1) / DEB_THREADS;
			debyePreVoltKernel<<<blocks, DEB_THREADS, 0, c.stream>>>(
				c.d_volt, c.local_dim, mg_cells[g], mg_relax[g], mg_drive[g],
				mg_W[g], mg_pre[g], mg_N[g], m_cuda_P);
		}
		return;
	}
	if (m_cuda_N == 0) return;
	int blocks = (m_cuda_N + DEB_THREADS - 1) / DEB_THREADS;
	debyePreVoltKernel<<<blocks, DEB_THREADS>>>(
		eng->GetDeviceVoltData(), eng->GetDeviceDimData(), d_cells,
		d_relax, d_drive, d_W, d_pre, m_cuda_N, m_cuda_P);
}

void Engine_Ext_DebyeMaterial::DoPostVoltageUpdatesCuda(Engine_cuda* eng)
{
	if (Engine_cuda_mgpu* mg = dynamic_cast<Engine_cuda_mgpu*>(eng)) {
		for (int g = 0; g < mg->NumSlabs(); ++g) {
			if ((int)mg_N.size() <= g || mg_N[g] == 0) continue;
			const CudaSlabCtx &c = mg->Slab(g);
			cudaSetDevice(c.device);
			int blocks = (mg_N[g] + DEB_THREADS - 1) / DEB_THREADS;
			debyePostVoltKernel<<<blocks, DEB_THREADS, 0, c.stream>>>(
				c.d_volt, c.local_dim, mg_cells[g], mg_drive[g], mg_W[g],
				mg_pre[g], mg_ade[g], mg_N[g], m_cuda_P);
		}
		return;
	}
	if (m_cuda_N == 0) return;
	int blocks = (m_cuda_N + DEB_THREADS - 1) / DEB_THREADS;
	debyePostVoltKernel<<<blocks, DEB_THREADS>>>(
		eng->GetDeviceVoltData(), eng->GetDeviceDimData(), d_cells,
		d_drive, d_W, d_pre, d_ade, m_cuda_N, m_cuda_P);
}

void Engine_Ext_DebyeMaterial::Apply2VoltagesCuda(Engine_cuda* eng)
{
	if (Engine_cuda_mgpu* mg = dynamic_cast<Engine_cuda_mgpu*>(eng)) {
		for (int g = 0; g < mg->NumSlabs(); ++g) {
			if ((int)mg_N.size() <= g || mg_N[g] == 0) continue;
			const CudaSlabCtx &c = mg->Slab(g);
			cudaSetDevice(c.device);
			int blocks = (mg_N[g] + DEB_THREADS - 1) / DEB_THREADS;
			debyeApplyKernel<<<blocks, DEB_THREADS, 0, c.stream>>>(
				c.d_volt, c.local_dim, mg_cells[g], mg_ade[g], mg_N[g]);
		}
		return;
	}
	if (m_cuda_N == 0) return;
	int blocks = (m_cuda_N + DEB_THREADS - 1) / DEB_THREADS;
	debyeApplyKernel<<<blocks, DEB_THREADS>>>(
		eng->GetDeviceVoltData(), eng->GetDeviceDimData(), d_cells, d_ade, m_cuda_N);
}

void Engine_Ext_DebyeMaterial::FreeCuda()
{
	FDTD_FLOAT** bufs[] = {&d_relax, &d_drive, &d_W, &d_pre, &d_ade};
	if (d_cells) { cudaFree(d_cells); d_cells = NULL; }
	for (FDTD_FLOAT** b : bufs)
		if (*b) { cudaFree(*b); *b = NULL; }
	m_cuda_N = 0;
	for (size_t g = 0; g < mg_dev.size(); ++g)
	{
		cudaSetDevice(mg_dev[g]);
		if (mg_cells[g]) cudaFree(mg_cells[g]);
		if (mg_relax[g]) cudaFree(mg_relax[g]);
		if (mg_drive[g]) cudaFree(mg_drive[g]);
		if (mg_W[g])     cudaFree(mg_W[g]);
		if (mg_pre[g])   cudaFree(mg_pre[g]);
		if (mg_ade[g])   cudaFree(mg_ade[g]);
	}
	mg_dev.clear(); mg_N.clear(); mg_cells.clear();
	mg_relax.clear(); mg_drive.clear(); mg_W.clear(); mg_pre.clear(); mg_ade.clear();
}
