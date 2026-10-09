/*
*	CUDA port of the invisible PML sheet (engine_ext_invisible_pml.cpp).
*
*	The virtual PML block, its coefficients and its fields live on the device,
*	and each of the four per-timestep hooks is one kernel, so the timestep stays
*	captured in the CUDA graph with no host-side field access. The kernels are
*	the host loops verbatim (same ranges, same stencil and lower-boundary shift,
*	same operation order per value), so the CUDA engine reproduces the CPU
*	engine to float rounding.
*
*	Multi-GPU: Operator_CUDA picks the slab cuts so that no sheet window is
*	split (see Operator_CUDA::ComputeSlabStarts). The whole block, the ghost
*	plane it reads and the sheet plane it writes then belong to one slab, and the
*	same kernels run there in slab-local coordinates on that slab's stream. The
*	sheet-plane write (post-voltage) and the normal-current write (apply-current)
*	both land before the engine's halo sends, so the neighbours see them.
*/

#include "engine_ext_invisible_pml.h"
#include "operator_ext_invisible_pml.h"
#include "FDTD/engine_cuda.h"
#include "FDTD/engine_cuda_mgpu.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include "hemi/grid_stride_range.h"
#include "tools/cuda/check.h"

#define IPML_THREADS 128

namespace {

struct IpmlGeom
{
	int L[3];			// virtual block size (local lines)
	int ny, nyP, nyPP;
	int lineInt;		// local index of the sheet plane
	int lineGhost;		// local index of the ghost (dual) layer
	int mainOfs[3];		// main-array index of local index 0 (transverse axes)
	int mainInt;		// main-array index of the sheet plane along ny
	int mainGhost;		// main-array index copied into the ghost layer
	dim3 mainDim;		// main (or slab-local) array dimensions
	int voltRange[2][2];
	int currRange[2][2];
};

__device__ __forceinline__ int locIdx(const IpmlGeom& g, int n, const int* p)
{
	return ((p[0] * g.L[1] + p[1]) * g.L[2] + p[2]) * 3 + n;
}

__device__ __forceinline__ int mainIdx(const IpmlGeom& g, int n, const int* pos)
{
	return ((pos[0] * (int)g.mainDim.y + pos[1]) * (int)g.mainDim.z + pos[2]) * 3 + n;
}

// DoPreVoltageUpdates: the main grid's tangential currents in front of the sheet -> ghost layer
__global__ void ipmlPreVoltKernel(IpmlGeom g, const FDTD_FLOAT* mainCurr, FDTD_FLOAT* curr)
{
	int W1 = g.L[g.nyPP];
	for (auto k : hemi::grid_stride_range(0, g.L[g.nyP] * W1))
	{
		int loc[3], pos[3];
		loc[g.nyP]  = k / W1;
		loc[g.nyPP] = k % W1;
		loc[g.ny]   = g.lineGhost;
		pos[g.nyP]  = loc[g.nyP]  + g.mainOfs[g.nyP];
		pos[g.nyPP] = loc[g.nyPP] + g.mainOfs[g.nyPP];
		pos[g.ny]   = g.mainGhost;
		curr[locIdx(g, g.nyP,  loc)] = mainCurr[mainIdx(g, g.nyP,  pos)];
		curr[locIdx(g, g.nyPP, loc)] = mainCurr[mainIdx(g, g.nyPP, pos)];
	}
}

// DoPostVoltageUpdates: UpdateVoltages over the block, then the sheet plane -> main grid.
// One thread per (component, local cell); a voltage depends only on currents, so
// writing the sheet-plane values right after computing them equals the host's
// "update everything, then copy" order.
__global__ void ipmlPostVoltKernel(IpmlGeom g,
	FDTD_FLOAT* volt, FDTD_FLOAT* volt_flux, const FDTD_FLOAT* curr,
	const FDTD_FLOAT* vv, const FDTD_FLOAT* vvfo, const FDTD_FLOAT* vvfn,
	const FDTD_FLOAT* vv_m, const FDTD_FLOAT* vi_m, FDTD_FLOAT* mainVolt)
{
	int nCells = g.L[0] * g.L[1] * g.L[2];
	for (auto t : hemi::grid_stride_range(0, 3 * nCells))
	{
		int n = t / nCells;
		int c = t % nCells;
		int p[3];
		p[2] = c % g.L[2];
		p[1] = (c / g.L[2]) % g.L[1];
		p[0] = c / (g.L[1] * g.L[2]);
		const int* range = g.voltRange[n == g.ny];
		if (p[g.ny] < range[0] || p[g.ny] > range[1])
			continue;

		int nP  = (n + 1) % 3;
		int nPP = (n + 2) % 3;
		int mP[3], mPP[3];
		for (int d = 0; d < 3; ++d)
			mP[d] = mPP[d] = p[d];
		mP[nP]   -= (p[nP] > 0);
		mPP[nPP] -= (p[nPP] > 0);

		int i = locIdx(g, n, p);
		FDTD_FLOAT f_help = vv[i] * volt[i] - vvfo[i] * volt_flux[i];
		FDTD_FLOAT flux = volt_flux[i];
		flux *= vv_m[i];
		flux += vi_m[i] * (
					curr[locIdx(g, nPP, p)] -
					curr[locIdx(g, nPP, mP)] -
					curr[locIdx(g, nP, p)] +
					curr[locIdx(g, nP, mPP)]
				);
		volt_flux[i] = flux;
		FDTD_FLOAT v = f_help + vvfn[i] * flux;
		volt[i] = v;

		// place the sheet plane into the main grid, where the PEC has zeroed it
		if (n != g.ny && p[g.ny] == g.lineInt)
		{
			int pos[3];
			pos[g.nyP]  = p[g.nyP]  + g.mainOfs[g.nyP];
			pos[g.nyPP] = p[g.nyPP] + g.mainOfs[g.nyPP];
			pos[g.ny]   = g.mainInt;
			mainVolt[mainIdx(g, n, pos)] = v;
		}
	}
}

// DoPreCurrentUpdates: the main grid owns the final sheet-plane voltages
__global__ void ipmlPreCurrKernel(IpmlGeom g, const FDTD_FLOAT* mainVolt, FDTD_FLOAT* volt)
{
	int W1 = g.L[g.nyPP];
	for (auto k : hemi::grid_stride_range(0, g.L[g.nyP] * W1))
	{
		int loc[3], pos[3];
		loc[g.nyP]  = k / W1;
		loc[g.nyPP] = k % W1;
		loc[g.ny]   = g.lineInt;
		pos[g.nyP]  = loc[g.nyP]  + g.mainOfs[g.nyP];
		pos[g.nyPP] = loc[g.nyPP] + g.mainOfs[g.nyPP];
		pos[g.ny]   = g.mainInt;
		volt[locIdx(g, g.nyP,  loc)] = mainVolt[mainIdx(g, g.nyP,  pos)];
		volt[locIdx(g, g.nyPP, loc)] = mainVolt[mainIdx(g, g.nyPP, pos)];
	}
}

// Apply2Current: UpdateCurrents over the block, then the normal current on the
// sheet plane -> main grid (only the sheet-plane voltages read it, but probes and
// dumps should see it). As in the main engine, the last transverse lines are not updated.
__global__ void ipmlApplyCurrKernel(IpmlGeom g,
	FDTD_FLOAT* curr, FDTD_FLOAT* curr_flux, const FDTD_FLOAT* volt,
	const FDTD_FLOAT* ii, const FDTD_FLOAT* iifo, const FDTD_FLOAT* iifn,
	const FDTD_FLOAT* ii_m, const FDTD_FLOAT* iv_m, FDTD_FLOAT* mainCurr)
{
	int nCells = g.L[0] * g.L[1] * g.L[2];
	for (auto t : hemi::grid_stride_range(0, 3 * nCells))
	{
		int n = t / nCells;
		int c = t % nCells;
		int p[3];
		p[2] = c % g.L[2];
		p[1] = (c / g.L[2]) % g.L[1];
		p[0] = c / (g.L[1] * g.L[2]);
		if (p[g.nyP] >= g.L[g.nyP] - 1 || p[g.nyPP] >= g.L[g.nyPP] - 1)
			continue;
		const int* range = g.currRange[n == g.ny];
		if (p[g.ny] < range[0] || p[g.ny] > range[1])
			continue;

		int nP  = (n + 1) % 3;
		int nPP = (n + 2) % 3;
		int pP[3], pPP[3];
		for (int d = 0; d < 3; ++d)
			pP[d] = pPP[d] = p[d];
		++pP[nP];
		++pPP[nPP];

		int i = locIdx(g, n, p);
		FDTD_FLOAT f_help = ii[i] * curr[i] - iifo[i] * curr_flux[i];
		FDTD_FLOAT flux = curr_flux[i];
		flux *= ii_m[i];
		flux += iv_m[i] * (
					volt[locIdx(g, nPP, p)] -
					volt[locIdx(g, nPP, pP)] -
					volt[locIdx(g, nP, p)] +
					volt[locIdx(g, nP, pPP)]
				);
		curr_flux[i] = flux;
		FDTD_FLOAT cv = f_help + iifn[i] * flux;
		curr[i] = cv;

		if (n == g.ny && p[g.ny] == g.lineInt)
		{
			int pos[3];
			pos[g.nyP]  = p[g.nyP]  + g.mainOfs[g.nyP];
			pos[g.nyPP] = p[g.nyPP] + g.mainOfs[g.nyPP];
			pos[g.ny]   = g.mainInt;
			mainCurr[mainIdx(g, n, pos)] = cv;
		}
	}
}

} // namespace

// The geometry the kernels need, rebuilt per launch from the extension's state
// (a handful of ints, passed by value -- graph capture records it).
#define IPML_GEOM(g)                                                     \
	IpmlGeom g;                                                          \
	for (int d = 0; d < 3; ++d) { g.L[d] = (int)m_numLines[d]; g.mainOfs[d] = m_mainOfs[d]; } \
	g.ny = m_ny; g.nyP = m_nyP; g.nyPP = m_nyPP;                         \
	g.lineInt = (int)m_lineInt; g.lineGhost = (int)m_lineGhost;          \
	g.mainInt = m_mainIntD; g.mainGhost = m_mainGhostD;                  \
	g.mainDim = m_mainDim;                                               \
	for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) {            \
		g.voltRange[a][b] = (int)m_voltRange[a][b];                      \
		g.currRange[a][b] = (int)m_currRange[a][b]; }

static FDTD_FLOAT* ipmlUpload(const ArrayLib::ArrayNIJK<FDTD_FLOAT>& a)
{
	FDTD_FLOAT* d = NULL;
	size_t bytes = (size_t)a.size() * sizeof(FDTD_FLOAT);
	checkCuda(cudaMalloc(&d, bytes));
	checkCuda(cudaMemcpy(d, a.data(), bytes, cudaMemcpyHostToDevice));
	return d;
}

static FDTD_FLOAT* ipmlZeros(size_t n)
{
	FDTD_FLOAT* d = NULL;
	checkCuda(cudaMalloc(&d, n * sizeof(FDTD_FLOAT)));
	checkCuda(cudaMemset(d, 0, n * sizeof(FDTD_FLOAT)));
	return d;
}

void Engine_Ext_InvisiblePML::SetEngine(Engine* eng)
{
	m_Eng = eng;
	if (eng->GetType() != Engine::CUDA)
		return;

	for (int d = 0; d < 3; ++d)
		m_mainOfs[d] = (int)m_offset[d];
	m_mainIntD   = (int)m_mainInt;
	m_mainGhostD = (int)m_mainGhost;

	if (Engine_cuda_mgpu* mg = dynamic_cast<Engine_cuda_mgpu*>(eng))
	{
		// find the slab that owns the block; along x the block spans either the
		// sheet/ghost planes (x-normal) or the window's x lines
		int xlo = (m_ny == 0) ? (int)std::min(m_mainInt, m_mainGhost) : (int)m_offset[0];
		int xhi = (m_ny == 0) ? (int)std::max(m_mainInt, m_mainGhost) : (int)(m_offset[0] + m_numLines[0] - 1);
		int owner = -1;
		for (int s = 0; s < mg->NumSlabs(); ++s)
		{
			const CudaSlabCtx& c = mg->Slab(s);
			if (xlo >= c.x_start && xhi < c.x_end) { owner = s; break; }
		}
		if (owner < 0)
			throw std::runtime_error("Invisible PML: the sheet window is split across GPU slabs "
			                         "(Operator_CUDA must place the cuts around it)");
		const CudaSlabCtx& c = mg->Slab(owner);
		m_dev = c.device;
		m_stream = c.stream;
		d_mainVolt = c.d_volt;
		d_mainCurr = c.d_curr;
		m_mainDim = c.local_dim;
		// slab-local x = global x - x_start + 1 (plane 0 is the left halo)
		if (m_ny == 0)
		{
			m_mainIntD   -= c.x_start - 1;
			m_mainGhostD -= c.x_start - 1;
		}
		else
			m_mainOfs[0] -= c.x_start - 1;
	}
	else
	{
		Engine_cuda* ce = static_cast<Engine_cuda*>(eng);
		checkCuda(cudaGetDevice(&m_dev));
		m_stream = cudaStreamPerThread;
		d_mainVolt = ce->GetDeviceVoltData();
		d_mainCurr = ce->GetDeviceCurrData();
		m_mainDim = ce->GetDeviceDimData();
	}

	checkCuda(cudaSetDevice(m_dev));
	const Operator_Ext_InvisiblePML* op = m_Op_PML;
	size_t n = (size_t)volt.size();
	d_volt      = ipmlZeros(n);
	d_volt_flux = ipmlZeros(n);
	d_curr      = ipmlZeros(n);
	d_curr_flux = ipmlZeros(n);
	d_vv_m = ipmlUpload(op->vv_m);  d_vi_m = ipmlUpload(op->vi_m);
	d_vv   = ipmlUpload(op->vv);    d_vvfo = ipmlUpload(op->vvfo);  d_vvfn = ipmlUpload(op->vvfn);
	d_ii_m = ipmlUpload(op->ii_m);  d_iv_m = ipmlUpload(op->iv_m);
	d_ii   = ipmlUpload(op->ii);    d_iifo = ipmlUpload(op->iifo);  d_iifn = ipmlUpload(op->iifn);
	m_cuda = true;
}

void Engine_Ext_InvisiblePML::FreeDevice()
{
	FDTD_FLOAT** all[] = {&d_volt, &d_volt_flux, &d_curr, &d_curr_flux,
		&d_vv_m, &d_vi_m, &d_vv, &d_vvfo, &d_vvfn, &d_ii_m, &d_iv_m, &d_ii, &d_iifo, &d_iifn};
	if (m_cuda)
		cudaSetDevice(m_dev);
	for (FDTD_FLOAT** p : all)
	{
		if (*p) cudaFree(*p);
		*p = NULL;
	}
	m_cuda = false;
}

static inline int ipmlBlocks(long long work)
{
	long long b = (work + IPML_THREADS - 1) / IPML_THREADS;
	return (int)(b < 1 ? 1 : (b > 65535 ? 65535 : b));
}

void Engine_Ext_InvisiblePML::DoPreVoltageUpdatesCuda()
{
	IPML_GEOM(g);
	cudaSetDevice(m_dev);
	ipmlPreVoltKernel<<<ipmlBlocks((long long)g.L[m_nyP] * g.L[m_nyPP]), IPML_THREADS, 0, m_stream>>>(
		g, d_mainCurr, d_curr);
}

void Engine_Ext_InvisiblePML::DoPostVoltageUpdatesCuda()
{
	IPML_GEOM(g);
	cudaSetDevice(m_dev);
	ipmlPostVoltKernel<<<ipmlBlocks(3LL * volt.size() / 3), IPML_THREADS, 0, m_stream>>>(
		g, d_volt, d_volt_flux, d_curr, d_vv, d_vvfo, d_vvfn, d_vv_m, d_vi_m, d_mainVolt);
}

void Engine_Ext_InvisiblePML::DoPreCurrentUpdatesCuda()
{
	IPML_GEOM(g);
	cudaSetDevice(m_dev);
	ipmlPreCurrKernel<<<ipmlBlocks((long long)g.L[m_nyP] * g.L[m_nyPP]), IPML_THREADS, 0, m_stream>>>(
		g, d_mainVolt, d_volt);
}

void Engine_Ext_InvisiblePML::Apply2CurrentCuda()
{
	IPML_GEOM(g);
	cudaSetDevice(m_dev);
	ipmlApplyCurrKernel<<<ipmlBlocks(3LL * volt.size() / 3), IPML_THREADS, 0, m_stream>>>(
		g, d_curr, d_curr_flux, d_volt, d_ii, d_iifo, d_iifn, d_ii_m, d_iv_m, d_mainCurr);
}
