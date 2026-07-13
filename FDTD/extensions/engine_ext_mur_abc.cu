/*
*	CUDA port of the first-order Mur ABC. Mirrors the host implementation in
*	engine_ext_mur_abc.cpp exactly, but as on-device kernels over the boundary
*	face so the whole per-timestep sequence stays captured in the CUDA graph.
*	No host-side field access: coeffs + aux boundary voltages live on the device.
*/

#include "engine_ext_mur_abc.h"
#include "operator_ext_mur_abc.h"
#include "FDTD/engine_cuda.h"

#include <cuda_runtime.h>
#include "hemi/grid_stride_range.h"
#include "tools/cuda/check.h"

#define MUR_THREADS 128

// Full-grid flat cell index for boundary face-cell (i,j) at line `lineNr` along
// the boundary-normal axis ny; matches Engine_cuda's FlatIndex(x,y,z).
__device__ __forceinline__
int mur_cell(int i, int j, int lineNr, int ny, int nyP, int nyPP, dim3 dim)
{
	int p[3];
	p[ny]   = lineNr;
	p[nyP]  = i;
	p[nyPP] = j;
	return p[0] * dim.y * dim.z + p[1] * dim.z + p[2];
}

// Pre: stash  E_interior^n - coeff*E_boundary^n   (before the core update)
__global__
void murPreVoltKernel(FDTD_FLOAT* volt, const FDTD_FLOAT* cP, const FDTD_FLOAT* cPP,
	FDTD_FLOAT* vP, FDTD_FLOAT* vPP, int W0, int W1, int ny, int nyP, int nyPP,
	int lineNr, int lineShift, dim3 dim, const int* d_numTS, int startTS)
{
	if (*d_numTS < startTS) return;
	for (auto k : hemi::grid_stride_range(0, W0 * W1)) {
		int i = k / W1, j = k % W1;
		int cell   = mur_cell(i, j, lineNr,    ny, nyP, nyPP, dim);
		int cell_s = mur_cell(i, j, lineShift, ny, nyP, nyPP, dim);
		vP[k]  = volt[cell_s * 3 + nyP]  - cP[k]  * volt[cell * 3 + nyP];
		vPP[k] = volt[cell_s * 3 + nyPP] - cPP[k] * volt[cell * 3 + nyPP];
	}
}

// Post: add coeff*E_interior^{n+1}  (after the core update)
__global__
void murPostVoltKernel(FDTD_FLOAT* volt, const FDTD_FLOAT* cP, const FDTD_FLOAT* cPP,
	FDTD_FLOAT* vP, FDTD_FLOAT* vPP, int W0, int W1, int ny, int nyP, int nyPP,
	int lineShift, dim3 dim, const int* d_numTS, int startTS)
{
	if (*d_numTS < startTS) return;
	for (auto k : hemi::grid_stride_range(0, W0 * W1)) {
		int i = k / W1, j = k % W1;
		int cell_s = mur_cell(i, j, lineShift, ny, nyP, nyPP, dim);
		vP[k]  += cP[k]  * volt[cell_s * 3 + nyP];
		vPP[k] += cPP[k] * volt[cell_s * 3 + nyPP];
	}
}

// Apply: write the absorbed boundary voltage back into the field
__global__
void murApplyVoltKernel(FDTD_FLOAT* volt, const FDTD_FLOAT* vP, const FDTD_FLOAT* vPP,
	int W0, int W1, int ny, int nyP, int nyPP, int lineNr, dim3 dim,
	const int* d_numTS, int startTS)
{
	if (*d_numTS < startTS) return;
	for (auto k : hemi::grid_stride_range(0, W0 * W1)) {
		int i = k / W1, j = k % W1;
		int cell = mur_cell(i, j, lineNr, ny, nyP, nyPP, dim);
		volt[cell * 3 + nyP]  = vP[k];
		volt[cell * 3 + nyPP] = vPP[k];
	}
}

void Engine_Ext_Mur_ABC::SetEngine(Engine* eng)
{
	m_Eng = eng;
	if (eng->GetType() != Engine::CUDA)
		return;

	int N = (int)(m_numLines[0] * m_numLines[1]);
	checkCuda(cudaMalloc(&d_mur_nyP,   N * sizeof(FDTD_FLOAT)));
	checkCuda(cudaMalloc(&d_mur_nyPP,  N * sizeof(FDTD_FLOAT)));
	checkCuda(cudaMalloc(&d_volt_nyP,  N * sizeof(FDTD_FLOAT)));
	checkCuda(cudaMalloc(&d_volt_nyPP, N * sizeof(FDTD_FLOAT)));
	// upload the per-face-cell Mur coefficients (row-major i*W1+j, matches ArrayIJ)
	checkCuda(cudaMemcpy(d_mur_nyP,  m_Mur_Coeff_nyP.data(),  N * sizeof(FDTD_FLOAT), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_mur_nyPP, m_Mur_Coeff_nyPP.data(), N * sizeof(FDTD_FLOAT), cudaMemcpyHostToDevice));
	checkCuda(cudaMemset(d_volt_nyP,  0, N * sizeof(FDTD_FLOAT)));
	checkCuda(cudaMemset(d_volt_nyPP, 0, N * sizeof(FDTD_FLOAT)));
}

void Engine_Ext_Mur_ABC::DoPreVoltageUpdatesCuda(Engine_cuda* eng)
{
	int N = (int)(m_numLines[0] * m_numLines[1]);
	int blocks = (N + MUR_THREADS - 1) / MUR_THREADS;
	murPreVoltKernel<<<blocks, MUR_THREADS>>>(
		eng->GetDeviceVoltData(), d_mur_nyP, d_mur_nyPP, d_volt_nyP, d_volt_nyPP,
		(int)m_numLines[0], (int)m_numLines[1], m_ny, m_nyP, m_nyPP,
		(int)m_LineNr, m_LineNr_Shift, eng->GetDeviceDimData(),
		eng->GetDeviceNumTS(), (int)m_start_TS);
}

void Engine_Ext_Mur_ABC::DoPostVoltageUpdatesCuda(Engine_cuda* eng)
{
	int N = (int)(m_numLines[0] * m_numLines[1]);
	int blocks = (N + MUR_THREADS - 1) / MUR_THREADS;
	murPostVoltKernel<<<blocks, MUR_THREADS>>>(
		eng->GetDeviceVoltData(), d_mur_nyP, d_mur_nyPP, d_volt_nyP, d_volt_nyPP,
		(int)m_numLines[0], (int)m_numLines[1], m_ny, m_nyP, m_nyPP,
		m_LineNr_Shift, eng->GetDeviceDimData(),
		eng->GetDeviceNumTS(), (int)m_start_TS);
}

void Engine_Ext_Mur_ABC::Apply2VoltagesCuda(Engine_cuda* eng)
{
	int N = (int)(m_numLines[0] * m_numLines[1]);
	int blocks = (N + MUR_THREADS - 1) / MUR_THREADS;
	murApplyVoltKernel<<<blocks, MUR_THREADS>>>(
		eng->GetDeviceVoltData(), d_volt_nyP, d_volt_nyPP,
		(int)m_numLines[0], (int)m_numLines[1], m_ny, m_nyP, m_nyPP,
		(int)m_LineNr, eng->GetDeviceDimData(),
		eng->GetDeviceNumTS(), (int)m_start_TS);
}
