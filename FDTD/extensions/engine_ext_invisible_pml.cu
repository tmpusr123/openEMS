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
#include <iostream>
#include <algorithm>
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
	const FDTD_FLOAT* vv_m, const FDTD_FLOAT* vi_m, FDTD_FLOAT* mainVolt, bool writeMain)
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
		if (writeMain && n != g.ny && p[g.ny] == g.lineInt)
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
	const FDTD_FLOAT* ii_m, const FDTD_FLOAT* iv_m, FDTD_FLOAT* mainCurr, bool writeMain)
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

		if (writeMain && n == g.ny && p[g.ny] == g.lineInt)
		{
			int pos[3];
			pos[g.nyP]  = p[g.nyP]  + g.mainOfs[g.nyP];
			pos[g.nyPP] = p[g.nyPP] + g.mainOfs[g.nyPP];
			pos[g.ny]   = g.mainInt;
			mainCurr[mainIdx(g, n, pos)] = cv;
		}
	}
}

// Split mode: one plane at fixed ny between a slab's main array and a contiguous
// buffer. x (axis 0) runs over [0, nx) from slab-local xLoc0; the other transverse
// axis T over [0, nT) from main index tOfs. Buffer layout ((ix*nT)+it)*nComp + c.
__global__ void ipmlMainPlaneKernel(FDTD_FLOAT* main, dim3 dim, int ny, int T, int line,
	int xLoc0, int nx, int nT, int tOfs, int c0, int c1, int nComp, FDTD_FLOAT* buf, bool toBuf)
{
	for (auto k : hemi::grid_stride_range(0, nx * nT))
	{
		int pos[3];
		pos[0]  = xLoc0 + k / nT;
		pos[T]  = tOfs + k % nT;
		pos[ny] = line;
		int cell = (pos[0] * (int)dim.y + pos[1]) * (int)dim.z + pos[2];
		for (int ci = 0; ci < nComp; ++ci)
		{
			int c = ci ? c1 : c0;
			if (toBuf) buf[k * nComp + ci] = main[cell * 3 + c];
			else       main[cell * 3 + c] = buf[k * nComp + ci];
		}
	}
}

// The same plane in the virtual block: local x from lx0, T from 0, ny = layer.
__global__ void ipmlVirtPlaneKernel(FDTD_FLOAT* virt, int L1, int L2, int ny, int T, int layer,
	int lx0, int nx, int nT, int c0, int c1, int nComp, FDTD_FLOAT* buf, bool toBuf)
{
	for (auto k : hemi::grid_stride_range(0, nx * nT))
	{
		int p[3];
		p[0]  = lx0 + k / nT;
		p[T]  = k % nT;
		p[ny] = layer;
		int cell = (p[0] * L1 + p[1]) * L2 + p[2];
		for (int ci = 0; ci < nComp; ++ci)
		{
			int c = ci ? c1 : c0;
			if (toBuf) buf[k * nComp + ci] = virt[cell * 3 + c];
			else       virt[cell * 3 + c] = buf[k * nComp + ci];
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
		{
			// the window crosses slab cuts (Operator_CUDA could not avoid it with
			// the slab count it wanted): run the block on one slab and exchange
			// the coupling planes each timestep
			if (m_ny == 0)
				throw std::runtime_error("Invisible PML: an x-normal sheet is split across GPU slabs "
				                         "(Operator_CUDA must place the cuts around it)");
			SetEngineSplit(mg);		// sets m_dev / m_stream to the owner slab
		}
		else
		{
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
	for (size_t i = 0; i < m_parts.size(); ++i)
	{
		cudaSetDevice(m_parts[i].dev);
		cudaFree(m_parts[i].d_buf);
		cudaEventDestroy(m_parts[i].evSlab);
		cudaSetDevice(m_dev);
		cudaFree(m_parts[i].d_stage);
		cudaEventDestroy(m_parts[i].evOwner);
	}
	m_parts.clear();
	m_split = false;
	m_cuda = false;
}

static inline int ipmlBlocks(long long work)
{
	long long b = (work + IPML_THREADS - 1) / IPML_THREADS;
	return (int)(b < 1 ? 1 : (b > 65535 ? 65535 : b));
}

void Engine_Ext_InvisiblePML::DoPreVoltageUpdatesCuda()
{
	if (m_split) { GatherPlane(true, (int)m_lineGhost, (int)m_mainGhost); return; }
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
		g, d_volt, d_volt_flux, d_curr, d_vv, d_vvfo, d_vvfn, d_vv_m, d_vi_m, d_mainVolt, !m_split);
	if (m_split) ScatterPlane(false, (int)m_lineInt, (int)m_mainInt);
}

void Engine_Ext_InvisiblePML::DoPreCurrentUpdatesCuda()
{
	if (m_split) { GatherPlane(false, (int)m_lineInt, (int)m_mainInt); return; }
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
		g, d_curr, d_curr_flux, d_volt, d_ii, d_iifo, d_iifn, d_ii_m, d_iv_m, d_mainCurr, !m_split);
	if (m_split) ScatterPlane(true, (int)m_lineInt, (int)m_mainInt);
}

// ---------------------------------------------------------------------------
// Multi-GPU split mode
// ---------------------------------------------------------------------------

static void ipmlCopy(FDTD_FLOAT* dst, int dstDev, const FDTD_FLOAT* src, int srcDev,
	size_t bytes, cudaStream_t stream)
{
	if (dstDev == srcDev)
		checkCuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream));
	else
		checkCuda(cudaMemcpyPeerAsync(dst, dstDev, src, srcDev, bytes, stream));
}

void Engine_Ext_InvisiblePML::SetEngineSplit(Engine_cuda_mgpu* mg)
{
	int X0 = (int)m_offset[0], X1 = X0 + (int)m_numLines[0];		// window x lines [X0, X1)
	int T = (m_nyP == 0) ? m_nyPP : m_nyP;
	int nT = (int)m_numLines[T];

	// owner: the slab holding most of the window
	int owner = 0, best = -1;
	for (int s = 0; s < mg->NumSlabs(); ++s)
	{
		const CudaSlabCtx& c = mg->Slab(s);
		int ov = std::min(X1, c.x_end) - std::max(X0, c.x_start);
		if (ov > best) { best = ov; owner = s; }
	}
	const CudaSlabCtx& co = mg->Slab(owner);
	m_dev = co.device;
	m_stream = co.stream;
	m_split = true;

	m_parts.clear();
	for (int s = 0; s < mg->NumSlabs(); ++s)
	{
		const CudaSlabCtx& c = mg->Slab(s);
		int x0 = std::max(X0, c.x_start), x1 = std::min(X1, c.x_end);
		if (x1 <= x0)
			continue;
		SplitPart p;
		p.slab = s; p.dev = c.device; p.stream = c.stream;
		p.d_mainVolt = c.d_volt; p.d_mainCurr = c.d_curr; p.mainDim = c.local_dim;
		p.xStartLocal = 1 - c.x_start;
		p.x0 = x0; p.x1 = x1;
		p.x1c = std::min(x1, X1 - 1);	// currents: the last transverse line is not updated
		size_t bytes = (size_t)(x1 - x0) * nT * 2 * sizeof(FDTD_FLOAT);
		checkCuda(cudaSetDevice(c.device));
		checkCuda(cudaMalloc(&p.d_buf, bytes));
		checkCuda(cudaEventCreateWithFlags(&p.evSlab, cudaEventDisableTiming));
		if (c.device != co.device)
		{
			cudaError_t e = cudaDeviceEnablePeerAccess(co.device, 0);
			if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
		}
		checkCuda(cudaSetDevice(co.device));
		checkCuda(cudaMalloc(&p.d_stage, bytes));
		checkCuda(cudaEventCreateWithFlags(&p.evOwner, cudaEventDisableTiming));
		if (c.device != co.device)
		{
			cudaError_t e = cudaDeviceEnablePeerAccess(c.device, 0);
			if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
		}
		m_parts.push_back(p);
	}
	std::cout << "Invisible PML: window x [" << X0 << "," << X1 << ") spans " << m_parts.size()
	          << " GPU slabs; virtual block on slab " << owner << ", coupling planes exchanged per timestep"
	          << std::endl;
}

// main grid plane (on each slab) -> virtual layer (on the owner)
void Engine_Ext_InvisiblePML::GatherPlane(bool curr, int lineLocal, int mainLine)
{
	int T = (m_nyP == 0) ? m_nyPP : m_nyP;
	int nT = (int)m_numLines[T];
	int X0 = (int)m_offset[0];
	for (size_t i = 0; i < m_parts.size(); ++i)
	{
		SplitPart& p = m_parts[i];
		int nx = p.x1 - p.x0;
		cudaSetDevice(p.dev);
		ipmlMainPlaneKernel<<<ipmlBlocks((long long)nx * nT), IPML_THREADS, 0, p.stream>>>(
			curr ? p.d_mainCurr : p.d_mainVolt, p.mainDim, m_ny, T, mainLine,
			p.x0 + p.xStartLocal, nx, nT, (int)m_offset[T], m_nyP, m_nyPP, 2, p.d_buf, true);
		ipmlCopy(p.d_stage, m_dev, p.d_buf, p.dev, (size_t)nx * nT * 2 * sizeof(FDTD_FLOAT), p.stream);
		checkCuda(cudaEventRecord(p.evSlab, p.stream));
	}
	cudaSetDevice(m_dev);
	for (size_t i = 0; i < m_parts.size(); ++i)
	{
		SplitPart& p = m_parts[i];
		int nx = p.x1 - p.x0;
		checkCuda(cudaStreamWaitEvent(m_stream, p.evSlab, 0));
		ipmlVirtPlaneKernel<<<ipmlBlocks((long long)nx * nT), IPML_THREADS, 0, m_stream>>>(
			curr ? d_curr : d_volt, (int)m_numLines[1], (int)m_numLines[2], m_ny, T, lineLocal,
			p.x0 - X0, nx, nT, m_nyP, m_nyPP, 2, p.d_stage, false);
	}
}

// virtual layer (on the owner) -> main grid plane (on each slab). Voltages: the
// tangential pair over the whole window; currents: the normal component, without
// the last transverse lines (as Apply2Current)
void Engine_Ext_InvisiblePML::ScatterPlane(bool curr, int lineLocal, int mainLine)
{
	int T = (m_nyP == 0) ? m_nyPP : m_nyP;
	int nT = (int)m_numLines[T] - (curr ? 1 : 0);
	int X0 = (int)m_offset[0];
	int nComp = curr ? 1 : 2;
	int c0 = curr ? m_ny : m_nyP, c1 = curr ? m_ny : m_nyPP;
	cudaSetDevice(m_dev);
	for (size_t i = 0; i < m_parts.size(); ++i)
	{
		SplitPart& p = m_parts[i];
		int nx = (curr ? p.x1c : p.x1) - p.x0;
		if (nx <= 0) continue;
		ipmlVirtPlaneKernel<<<ipmlBlocks((long long)nx * nT), IPML_THREADS, 0, m_stream>>>(
			curr ? d_curr : d_volt, (int)m_numLines[1], (int)m_numLines[2], m_ny, T, lineLocal,
			p.x0 - X0, nx, nT, c0, c1, nComp, p.d_stage, true);
		ipmlCopy(p.d_buf, p.dev, p.d_stage, m_dev, (size_t)nx * nT * nComp * sizeof(FDTD_FLOAT), m_stream);
		checkCuda(cudaEventRecord(p.evOwner, m_stream));
	}
	for (size_t i = 0; i < m_parts.size(); ++i)
	{
		SplitPart& p = m_parts[i];
		int nx = (curr ? p.x1c : p.x1) - p.x0;
		if (nx <= 0) continue;
		cudaSetDevice(p.dev);
		checkCuda(cudaStreamWaitEvent(p.stream, p.evOwner, 0));
		ipmlMainPlaneKernel<<<ipmlBlocks((long long)nx * nT), IPML_THREADS, 0, p.stream>>>(
			curr ? p.d_mainCurr : p.d_mainVolt, p.mainDim, m_ny, T, mainLine,
			p.x0 + p.xStartLocal, nx, nT, (int)m_offset[T], c0, c1, nComp, p.d_buf, false);
	}
}
