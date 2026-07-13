/*
*	CUDA port of the Drude/Lorentz/Debye dispersive-material ADE extension.
*	Mirrors the host recurrence in engine_ext_lorentzmaterial.cpp (voltage and
*	current pre-update ADE integration) and engine_ext_dispersive.cpp (the
*	Apply2* subtraction), but as on-device kernels so the whole per-timestep
*	sequence is captured into the CUDA graph. No host-side field access.
*
*	The host arrays are [order][field_component][cell]; here each (order,cell)
*	is one "disp_cell" carrying the three field-component coefficients, and the
*	ADE aux state lives in flat device arrays indexed (cell*3 + component). One
*	kernel is launched per dispersion order so that, for multi-pole materials
*	where the same cell appears in several orders, the field write-backs stay
*	serialized exactly like the host's outer order loop (no atomics needed).
*/

#include "engine_ext_lorentzmaterial.h"
#include "operator_ext_lorentzmaterial.h"
#include "FDTD/engine_cuda.h"

#include <cuda_runtime.h>
#include "hemi/grid_stride_range.h"
#include "tools/cuda/check.h"

#define DISP_THREADS 128

typedef Engine_Ext_LorentzMaterial::disp_cell disp_cell;

// Pre-voltage: integrate the voltage ADE using the pre-core-update node voltage.
// Matches Engine_Ext_LorentzMaterial::DoPreVoltageUpdatesImpl.
__global__
void lorPreVoltKernel(const FDTD_FLOAT* volt, dim3 dim, const disp_cell* c,
	FDTD_FLOAT* vADE, FDTD_FLOAT* vLor, int N, int lorOn)
{
	for (auto k : hemi::grid_stride_range(0, N)) {
		disp_cell e = c[k];
		int base = (e.x * dim.y * dim.z + e.y * dim.z + e.z) * 3;
		#pragma unroll
		for (int n = 0; n < 3; ++n) {
			FDTD_FLOAT Veng = volt[base + n];
			FDTD_FLOAT a = vADE[k * 3 + n];
			if (lorOn) {
				FDTD_FLOAT L = vLor[k * 3 + n] + e.v_lor[n] * a;   // uses old ADE
				vLor[k * 3 + n] = L;
				vADE[k * 3 + n] = a * e.v_int[n] + e.v_ext[n] * (Veng - L);
			} else {
				vADE[k * 3 + n] = a * e.v_int[n] + e.v_ext[n] * Veng;
			}
		}
	}
}

// Pre-current: integrate the current ADE using the pre-core-update node current.
// Matches Engine_Ext_LorentzMaterial::DoPreCurrentUpdatesImpl.
__global__
void lorPreCurrKernel(const FDTD_FLOAT* curr, dim3 dim, const disp_cell* c,
	FDTD_FLOAT* iADE, FDTD_FLOAT* iLor, int N, int lorOn)
{
	for (auto k : hemi::grid_stride_range(0, N)) {
		disp_cell e = c[k];
		int base = (e.x * dim.y * dim.z + e.y * dim.z + e.z) * 3;
		#pragma unroll
		for (int n = 0; n < 3; ++n) {
			FDTD_FLOAT Ceng = curr[base + n];
			FDTD_FLOAT a = iADE[k * 3 + n];
			if (lorOn) {
				FDTD_FLOAT L = iLor[k * 3 + n] + e.i_lor[n] * a;   // uses old ADE
				iLor[k * 3 + n] = L;
				iADE[k * 3 + n] = a * e.i_int[n] + e.i_ext[n] * (Ceng - L);
			} else {
				iADE[k * 3 + n] = a * e.i_int[n] + e.i_ext[n] * Ceng;
			}
		}
	}
}

// Apply: subtract the integrated ADE from the freshly core-updated field.
// Matches Engine_Ext_Dispersive::Apply2VoltagesImpl / Apply2CurrentImpl.
__global__
void dispApplyKernel(FDTD_FLOAT* field, dim3 dim, const disp_cell* c,
	const FDTD_FLOAT* ADE, int N)
{
	for (auto k : hemi::grid_stride_range(0, N)) {
		disp_cell e = c[k];
		int base = (e.x * dim.y * dim.z + e.y * dim.z + e.z) * 3;
		field[base + 0] -= ADE[k * 3 + 0];
		field[base + 1] -= ADE[k * 3 + 1];
		field[base + 2] -= ADE[k * 3 + 2];
	}
}

void Engine_Ext_LorentzMaterial::SetEngine(Engine* eng)
{
	m_Eng = eng;
	if (eng->GetType() != Engine::CUDA)
		return;

	Operator_Ext_LorentzMaterial* op = m_Op_Ext_Lor;
	m_cuda_order = m_Order;

	m_cuda_N.assign(m_Order, 0);
	m_cuda_vOn.assign(m_Order, 0);
	m_cuda_vLorOn.assign(m_Order, 0);
	m_cuda_iOn.assign(m_Order, 0);
	m_cuda_iLorOn.assign(m_Order, 0);
	d_cells.assign(m_Order, NULL);
	d_vADE.assign(m_Order, NULL);
	d_vLor.assign(m_Order, NULL);
	d_iADE.assign(m_Order, NULL);
	d_iLor.assign(m_Order, NULL);

	for (int o = 0; o < m_Order; ++o)
	{
		int N = (int)op->m_LM_Count[o];
		bool vOn  = op->m_volt_ADE_On[o];
		bool iOn  = op->m_curr_ADE_On[o];
		bool vLor = op->m_volt_Lor_ADE_On[o];
		bool iLor = op->m_curr_Lor_ADE_On[o];

		m_cuda_N[o]      = N;
		m_cuda_vOn[o]    = vOn;
		m_cuda_iOn[o]    = iOn;
		m_cuda_vLorOn[o] = vLor;
		m_cuda_iLorOn[o] = iLor;

		if (N == 0)
			continue;

		// pack geometry + coefficients AoS; off-side coeffs stay zero (unused)
		disp_cell* h = new disp_cell[N];
		unsigned int** pos = op->m_LM_pos[o];
		for (int i = 0; i < N; ++i)
		{
			h[i].x = (int)pos[0][i];
			h[i].y = (int)pos[1][i];
			h[i].z = (int)pos[2][i];
			for (int n = 0; n < 3; ++n)
			{
				h[i].v_int[n] = vOn  ? op->v_int_ADE[o][n][i] : 0;
				h[i].v_ext[n] = vOn  ? op->v_ext_ADE[o][n][i] : 0;
				h[i].v_lor[n] = vLor ? op->v_Lor_ADE[o][n][i] : 0;
				h[i].i_int[n] = iOn  ? op->i_int_ADE[o][n][i] : 0;
				h[i].i_ext[n] = iOn  ? op->i_ext_ADE[o][n][i] : 0;
				h[i].i_lor[n] = iLor ? op->i_Lor_ADE[o][n][i] : 0;
			}
		}
		checkCuda(cudaMalloc(&d_cells[o], N * sizeof(disp_cell)));
		checkCuda(cudaMemcpy(d_cells[o], h, N * sizeof(disp_cell), cudaMemcpyHostToDevice));
		delete[] h;

		// ADE aux state (N*3), zero-initialised to match the host engine ctor
		size_t bytes = (size_t)N * 3 * sizeof(FDTD_FLOAT);
		if (vOn)  { checkCuda(cudaMalloc(&d_vADE[o], bytes)); checkCuda(cudaMemset(d_vADE[o], 0, bytes)); }
		if (vLor) { checkCuda(cudaMalloc(&d_vLor[o], bytes)); checkCuda(cudaMemset(d_vLor[o], 0, bytes)); }
		if (iOn)  { checkCuda(cudaMalloc(&d_iADE[o], bytes)); checkCuda(cudaMemset(d_iADE[o], 0, bytes)); }
		if (iLor) { checkCuda(cudaMalloc(&d_iLor[o], bytes)); checkCuda(cudaMemset(d_iLor[o], 0, bytes)); }
	}
}

void Engine_Ext_LorentzMaterial::DoPreVoltageUpdatesCuda(Engine_cuda* eng)
{
	for (int o = 0; o < m_cuda_order; ++o)
	{
		if (!m_cuda_vOn[o] || m_cuda_N[o] == 0) continue;
		int N = m_cuda_N[o];
		int blocks = (N + DISP_THREADS - 1) / DISP_THREADS;
		lorPreVoltKernel<<<blocks, DISP_THREADS>>>(
			eng->GetDeviceVoltData(), eng->GetDeviceDimData(),
			d_cells[o], d_vADE[o], d_vLor[o], N, (int)m_cuda_vLorOn[o]);
	}
}

void Engine_Ext_LorentzMaterial::Apply2VoltagesCuda(Engine_cuda* eng)
{
	for (int o = 0; o < m_cuda_order; ++o)
	{
		if (!m_cuda_vOn[o] || m_cuda_N[o] == 0) continue;
		int N = m_cuda_N[o];
		int blocks = (N + DISP_THREADS - 1) / DISP_THREADS;
		dispApplyKernel<<<blocks, DISP_THREADS>>>(
			eng->GetDeviceVoltData(), eng->GetDeviceDimData(),
			d_cells[o], d_vADE[o], N);
	}
}

void Engine_Ext_LorentzMaterial::DoPreCurrentUpdatesCuda(Engine_cuda* eng)
{
	for (int o = 0; o < m_cuda_order; ++o)
	{
		if (!m_cuda_iOn[o] || m_cuda_N[o] == 0) continue;
		int N = m_cuda_N[o];
		int blocks = (N + DISP_THREADS - 1) / DISP_THREADS;
		lorPreCurrKernel<<<blocks, DISP_THREADS>>>(
			eng->GetDeviceCurrData(), eng->GetDeviceDimData(),
			d_cells[o], d_iADE[o], d_iLor[o], N, (int)m_cuda_iLorOn[o]);
	}
}

void Engine_Ext_LorentzMaterial::Apply2CurrentCuda(Engine_cuda* eng)
{
	for (int o = 0; o < m_cuda_order; ++o)
	{
		if (!m_cuda_iOn[o] || m_cuda_N[o] == 0) continue;
		int N = m_cuda_N[o];
		int blocks = (N + DISP_THREADS - 1) / DISP_THREADS;
		dispApplyKernel<<<blocks, DISP_THREADS>>>(
			eng->GetDeviceCurrData(), eng->GetDeviceDimData(),
			d_cells[o], d_iADE[o], N);
	}
}
