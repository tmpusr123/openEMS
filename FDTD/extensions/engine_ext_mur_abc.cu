/*
*	CUDA port of the first-order Mur ABC. Mirrors the host implementation in
*	engine_ext_mur_abc.cpp exactly, but as on-device kernels over the boundary
*	face so the whole per-timestep sequence stays captured in the CUDA graph.
*	No host-side field access: coeffs + aux boundary voltages live on the device.
*/

#include "engine_ext_mur_abc.h"
#include "operator_ext_mur_abc.h"
#include "FDTD/engine_cuda.h"
#include "FDTD/engine_cuda_mgpu.h"

#include <cuda_runtime.h>
#include "hemi/grid_stride_range.h"
#include "tools/cuda/check.h"

#define MUR_THREADS 128

// Flat cell index for boundary face-cell (i,j) at line `lineNr` along the
// boundary-normal axis ny; matches Engine_cuda's FlatIndex(x,y,z). iOfs/jOfs
// shift the face coordinates into slab-local space in multi-GPU mode
// (0 on the single-GPU path).
__device__ __forceinline__
int mur_cell(int i, int j, int lineNr, int ny, int nyP, int nyPP, dim3 dim,
	int iOfs, int jOfs)
{
	int p[3];
	p[ny]   = lineNr;
	p[nyP]  = i + iOfs;
	p[nyPP] = j + jOfs;
	return p[0] * dim.y * dim.z + p[1] * dim.z + p[2];
}

// Pre: stash  E_interior^n - coeff*E_boundary^n   (before the core update)
__global__
void murPreVoltKernel(FDTD_FLOAT* volt, const FDTD_FLOAT* cP, const FDTD_FLOAT* cPP,
	FDTD_FLOAT* vP, FDTD_FLOAT* vPP, int W0, int W1, int ny, int nyP, int nyPP,
	int lineNr, int lineShift, dim3 dim, const int* d_numTS, int startTS,
	int iOfs, int jOfs)
{
	if (*d_numTS < startTS) return;
	for (auto k : hemi::grid_stride_range(0, W0 * W1)) {
		int i = k / W1, j = k % W1;
		int cell   = mur_cell(i, j, lineNr,    ny, nyP, nyPP, dim, iOfs, jOfs);
		int cell_s = mur_cell(i, j, lineShift, ny, nyP, nyPP, dim, iOfs, jOfs);
		vP[k]  = volt[cell_s * 3 + nyP]  - cP[k]  * volt[cell * 3 + nyP];
		vPP[k] = volt[cell_s * 3 + nyPP] - cPP[k] * volt[cell * 3 + nyPP];
	}
}

// Post: add coeff*E_interior^{n+1}  (after the core update)
__global__
void murPostVoltKernel(FDTD_FLOAT* volt, const FDTD_FLOAT* cP, const FDTD_FLOAT* cPP,
	FDTD_FLOAT* vP, FDTD_FLOAT* vPP, int W0, int W1, int ny, int nyP, int nyPP,
	int lineShift, dim3 dim, const int* d_numTS, int startTS, int iOfs, int jOfs)
{
	if (*d_numTS < startTS) return;
	for (auto k : hemi::grid_stride_range(0, W0 * W1)) {
		int i = k / W1, j = k % W1;
		int cell_s = mur_cell(i, j, lineShift, ny, nyP, nyPP, dim, iOfs, jOfs);
		vP[k]  += cP[k]  * volt[cell_s * 3 + nyP];
		vPP[k] += cPP[k] * volt[cell_s * 3 + nyPP];
	}
}

// Apply: write the absorbed boundary voltage back into the field
__global__
void murApplyVoltKernel(FDTD_FLOAT* volt, const FDTD_FLOAT* vP, const FDTD_FLOAT* vPP,
	int W0, int W1, int ny, int nyP, int nyPP, int lineNr, dim3 dim,
	const int* d_numTS, int startTS, int iOfs, int jOfs)
{
	if (*d_numTS < startTS) return;
	for (auto k : hemi::grid_stride_range(0, W0 * W1)) {
		int i = k / W1, j = k % W1;
		int cell = mur_cell(i, j, lineNr, ny, nyP, nyPP, dim, iOfs, jOfs);
		volt[cell * 3 + nyP]  = vP[k];
		volt[cell * 3 + nyPP] = vPP[k];
	}
}

// ---- multi-GPU: clip this face to each slab's x-range ----
void Engine_Ext_Mur_ABC::SetEngineMg(Engine_cuda_mgpu* mg)
{
	m_mgpu = true;
	int nslab = mg->NumSlabs();
	int W0 = (int)m_numLines[0], W1 = (int)m_numLines[1];
	const FDTD_FLOAT* hP  = m_Mur_Coeff_nyP.data();
	const FDTD_FLOAT* hPP = m_Mur_Coeff_nyPP.data();

	mg_dev.assign(nslab, 0); mg_W0.assign(nslab, 0); mg_W1.assign(nslab, 0);
	mg_iofs.assign(nslab, 0); mg_jofs.assign(nslab, 0);
	mg_line.assign(nslab, 0); mg_shift.assign(nslab, 0);
	mg_cP.assign(nslab, NULL); mg_cPP.assign(nslab, NULL);
	mg_vP.assign(nslab, NULL); mg_vPP.assign(nslab, NULL);

	for (int g = 0; g < nslab; ++g)
	{
		const CudaSlabCtx &c = mg->Slab(g);
		mg_dev[g] = c.device;

		int w0 = 0, w1 = 0, i0 = 0, j0 = 0;   // clipped dims + source offsets
		int iofs = 0, jofs = 0;
		int line = (int)m_LineNr, shift = m_LineNr_Shift;

		if (m_ny == 0)
		{
			// x-normal face: lives wholly on the slab that owns its x line
			// (the inner line is adjacent, so it is in the same slab).
			if ((int)m_LineNr < c.x_start || (int)m_LineNr >= c.x_end) continue;
			w0 = W0; w1 = W1;
			line  = (int)m_LineNr    - c.x_start + 1;
			shift = m_LineNr_Shift   - c.x_start + 1;
		}
		else if (m_nyP == 0)
		{
			// face i-axis is x (ny==2): clip rows [x_start, x_end)
			i0 = c.x_start; w0 = c.x_end - c.x_start; w1 = W1;
			iofs = 1 - 0;   // local x = (i - i0) + i0 - x_start + 1 = i_rel + 1
		}
		else
		{
			// face j-axis is x (ny==1): clip columns [x_start, x_end)
			j0 = c.x_start; w1 = c.x_end - c.x_start; w0 = W0;
			jofs = 1;
		}
		if (w0 <= 0 || w1 <= 0) continue;

		mg_W0[g] = w0; mg_W1[g] = w1;
		mg_iofs[g] = iofs; mg_jofs[g] = jofs;
		mg_line[g] = line; mg_shift[g] = shift;

		// slice the coeff arrays (row-major i*W1+j in the source)
		std::vector<FDTD_FLOAT> sP((size_t)w0 * w1), sPP((size_t)w0 * w1);
		for (int i = 0; i < w0; ++i)
			for (int j = 0; j < w1; ++j)
			{
				sP [(size_t)i * w1 + j] = hP [(size_t)(i + i0) * W1 + (j + j0)];
				sPP[(size_t)i * w1 + j] = hPP[(size_t)(i + i0) * W1 + (j + j0)];
			}

		checkCuda(cudaSetDevice(c.device));
		size_t bytes = (size_t)w0 * w1 * sizeof(FDTD_FLOAT);
		checkCuda(cudaMalloc(&mg_cP[g],  bytes));
		checkCuda(cudaMalloc(&mg_cPP[g], bytes));
		checkCuda(cudaMemcpy(mg_cP[g],  sP.data(),  bytes, cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(mg_cPP[g], sPP.data(), bytes, cudaMemcpyHostToDevice));
		checkCuda(cudaMalloc(&mg_vP[g],  bytes));
		checkCuda(cudaMalloc(&mg_vPP[g], bytes));
		checkCuda(cudaMemset(mg_vP[g],  0, bytes));
		checkCuda(cudaMemset(mg_vPP[g], 0, bytes));
	}
}

void Engine_Ext_Mur_ABC::DoPreVoltageUpdatesMg(Engine_cuda_mgpu* mg)
{
	for (int g = 0; g < mg->NumSlabs(); ++g)
	{
		if (mg_W0[g] == 0) continue;
		const CudaSlabCtx &c = mg->Slab(g);
		cudaSetDevice(c.device);
		int N = mg_W0[g] * mg_W1[g];
		int blocks = (N + MUR_THREADS - 1) / MUR_THREADS;
		murPreVoltKernel<<<blocks, MUR_THREADS, 0, c.stream>>>(
			c.d_volt, mg_cP[g], mg_cPP[g], mg_vP[g], mg_vPP[g],
			mg_W0[g], mg_W1[g], m_ny, m_nyP, m_nyPP,
			mg_line[g], mg_shift[g], c.local_dim, c.d_numTS, (int)m_start_TS,
			mg_iofs[g], mg_jofs[g]);
	}
}

void Engine_Ext_Mur_ABC::DoPostVoltageUpdatesMg(Engine_cuda_mgpu* mg)
{
	for (int g = 0; g < mg->NumSlabs(); ++g)
	{
		if (mg_W0[g] == 0) continue;
		const CudaSlabCtx &c = mg->Slab(g);
		cudaSetDevice(c.device);
		int N = mg_W0[g] * mg_W1[g];
		int blocks = (N + MUR_THREADS - 1) / MUR_THREADS;
		murPostVoltKernel<<<blocks, MUR_THREADS, 0, c.stream>>>(
			c.d_volt, mg_cP[g], mg_cPP[g], mg_vP[g], mg_vPP[g],
			mg_W0[g], mg_W1[g], m_ny, m_nyP, m_nyPP,
			mg_shift[g], c.local_dim, c.d_numTS, (int)m_start_TS,
			mg_iofs[g], mg_jofs[g]);
	}
}

void Engine_Ext_Mur_ABC::Apply2VoltagesMg(Engine_cuda_mgpu* mg)
{
	for (int g = 0; g < mg->NumSlabs(); ++g)
	{
		if (mg_W0[g] == 0) continue;
		const CudaSlabCtx &c = mg->Slab(g);
		cudaSetDevice(c.device);
		int N = mg_W0[g] * mg_W1[g];
		int blocks = (N + MUR_THREADS - 1) / MUR_THREADS;
		murApplyVoltKernel<<<blocks, MUR_THREADS, 0, c.stream>>>(
			c.d_volt, mg_vP[g], mg_vPP[g],
			mg_W0[g], mg_W1[g], m_ny, m_nyP, m_nyPP,
			mg_line[g], c.local_dim, c.d_numTS, (int)m_start_TS,
			mg_iofs[g], mg_jofs[g]);
	}
}

void Engine_Ext_Mur_ABC::SetEngine(Engine* eng)
{
	m_Eng = eng;
	if (eng->GetType() != Engine::CUDA)
		return;

	if (Engine_cuda_mgpu* mg = dynamic_cast<Engine_cuda_mgpu*>(eng)) {
		SetEngineMg(mg);
		return;
	}

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
	if (m_mgpu) { DoPreVoltageUpdatesMg(static_cast<Engine_cuda_mgpu*>(m_Eng)); return; }
	int N = (int)(m_numLines[0] * m_numLines[1]);
	int blocks = (N + MUR_THREADS - 1) / MUR_THREADS;
	murPreVoltKernel<<<blocks, MUR_THREADS>>>(
		eng->GetDeviceVoltData(), d_mur_nyP, d_mur_nyPP, d_volt_nyP, d_volt_nyPP,
		(int)m_numLines[0], (int)m_numLines[1], m_ny, m_nyP, m_nyPP,
		(int)m_LineNr, m_LineNr_Shift, eng->GetDeviceDimData(),
		eng->GetDeviceNumTS(), (int)m_start_TS, 0, 0);
}

void Engine_Ext_Mur_ABC::DoPostVoltageUpdatesCuda(Engine_cuda* eng)
{
	if (m_mgpu) { DoPostVoltageUpdatesMg(static_cast<Engine_cuda_mgpu*>(m_Eng)); return; }
	int N = (int)(m_numLines[0] * m_numLines[1]);
	int blocks = (N + MUR_THREADS - 1) / MUR_THREADS;
	murPostVoltKernel<<<blocks, MUR_THREADS>>>(
		eng->GetDeviceVoltData(), d_mur_nyP, d_mur_nyPP, d_volt_nyP, d_volt_nyPP,
		(int)m_numLines[0], (int)m_numLines[1], m_ny, m_nyP, m_nyPP,
		m_LineNr_Shift, eng->GetDeviceDimData(),
		eng->GetDeviceNumTS(), (int)m_start_TS, 0, 0);
}

void Engine_Ext_Mur_ABC::Apply2VoltagesCuda(Engine_cuda* eng)
{
	if (m_mgpu) { Apply2VoltagesMg(static_cast<Engine_cuda_mgpu*>(m_Eng)); return; }
	int N = (int)(m_numLines[0] * m_numLines[1]);
	int blocks = (N + MUR_THREADS - 1) / MUR_THREADS;
	murApplyVoltKernel<<<blocks, MUR_THREADS>>>(
		eng->GetDeviceVoltData(), d_volt_nyP, d_volt_nyPP,
		(int)m_numLines[0], (int)m_numLines[1], m_ny, m_nyP, m_nyPP,
		(int)m_LineNr, eng->GetDeviceDimData(),
		eng->GetDeviceNumTS(), (int)m_start_TS, 0, 0);
}
