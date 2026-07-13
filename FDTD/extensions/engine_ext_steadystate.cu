/*
*	CUDA port of the steady-state detection extension. On the CUDA engine the
*	whole timestep is captured into a graph, so the host-side per-timestep
*	recording in engine_ext_steadystate.cpp would run only once (at capture) and
*	never again. Here the probe voltages are recorded into a device ring buffer
*	every timestep, and the period convergence metric (whole-grid energy diff +
*	per-probe power diff, exactly as the host computes) is evaluated on-device at
*	each period boundary. GetLastDiff() reads the scalar result back.
*
*	This extension modifies no fields — it only observes them to decide when a
*	periodic (CW) excitation has reached steady state.
*/

#include "engine_ext_steadystate.h"
#include "operator_ext_steadystate.h"
#include "FDTD/engine_cuda.h"

#include <cuda_runtime.h>
#include "hemi/grid_stride_range.h"   // brings in hemi's checkCuda()
#include "tools/cuda/check.h"
#include "tools/constants.h"          // __EPS0__ / __MUE0__

typedef Engine_Ext_SteadyState::ss_probe ss_probe;

// Record each probe's node voltage into the ring buffer slot for this timestep.
// Matches the host loop in Apply2Voltages (rel_pos = numTS % 2p).
__global__
void ssRecordKernel(const FDTD_FLOAT* volt, dim3 dim, const ss_probe* probes,
	FDTD_FLOAT* records, int n_probes, int p, const int* d_numTS)
{
	int n = threadIdx.x + blockIdx.x * blockDim.x;
	if (n >= n_probes) return;
	int rel_pos = (*d_numTS) % (2 * p);
	ss_probe pr = probes[n];
	int flat = (pr.x * dim.y * dim.z + pr.y * dim.z + pr.z) * 3 + pr.dir;
	records[(long long)n * 2 * p + rel_pos] = volt[flat];
}

// At each period boundary (numTS % p == 0, numTS >= 2p) compute the convergence
// metric: max(whole-grid energy relative change, per-probe power relative diff).
// Single block; the energy reduction is grid-strided across the block.
__global__
void ssMetricKernel(const FDTD_FLOAT* volt, const FDTD_FLOAT* curr, dim3 dim,
	const FDTD_FLOAT* records, int n_probes, int p, const int* d_numTS,
	double* d_last_energy, double* d_last_diff, int* d_valid)
{
	int numTS = *d_numTS;
	if (numTS % p != 0 || numTS < 2 * p) return;   // only at period boundaries

	// whole-grid energy, matching Engine_Interface_FDTD::CalcFastEnergy exactly:
	// E and H summed separately over cells with x,y,z < numLines-1 (last layer
	// excluded), then weighted EPS0*E + MUE0*H. Single-block grid-stride reduce.
	long long dydz  = (long long)dim.y * dim.z;
	long long Ncells = (long long)dim.x * dydz;
	__shared__ double shE[256];
	__shared__ double shH[256];
	double accE = 0.0, accH = 0.0;
	for (long long cell = threadIdx.x; cell < Ncells; cell += blockDim.x) {
		int x = (int)(cell / dydz);
		long long rem = cell - (long long)x * dydz;
		int y = (int)(rem / dim.z);
		int z = (int)(rem - (long long)y * dim.z);
		if (x == (int)dim.x - 1 || y == (int)dim.y - 1 || z == (int)dim.z - 1)
			continue;
		long long base = cell * 3;
		double v0 = volt[base], v1 = volt[base + 1], v2 = volt[base + 2];
		double c0 = curr[base], c1 = curr[base + 1], c2 = curr[base + 2];
		accE += v0 * v0 + v1 * v1 + v2 * v2;
		accH += c0 * c0 + c1 * c1 + c2 * c2;
	}
	shE[threadIdx.x] = accE;
	shH[threadIdx.x] = accH;
	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			shE[threadIdx.x] += shE[threadIdx.x + s];
			shH[threadIdx.x] += shH[threadIdx.x + s];
		}
		__syncthreads();
	}
	if (threadIdx.x != 0) return;

	double curr_E = __EPS0__ * shE[0] + __MUE0__ * shH[0];

	int rel_pos = numTS % (2 * p);
	int new_pos = p, old_pos = 0;
	if (rel_pos <= p) { new_pos = 0; old_pos = p; }

	double last_max_diff = 0.0;
	int no_valid = 1;
	double last_E = *d_last_energy;
	if (last_E > 0.0) { last_max_diff = fabs(curr_E - last_E) / last_E; no_valid = 0; }
	*d_last_energy = curr_E;

	// max period power over all probes -> significance threshold
	double max_pow = 0.0;
	for (int n = 0; n < n_probes; ++n) {
		const FDTD_FLOAT* buf = records + (long long)n * 2 * p;
		double cp = 0.0;
		for (int nt = 0; nt < p; ++nt) { double b = buf[nt + new_pos]; cp += b * b; }
		if (cp > max_pow) max_pow = cp;
	}
	for (int n = 0; n < n_probes; ++n) {
		const FDTD_FLOAT* buf = records + (long long)n * 2 * p;
		double cp = 0.0, dp = 0.0;
		for (int nt = 0; nt < p; ++nt) {
			double bn = buf[nt + new_pos], bo = buf[nt + old_pos];
			cp += bn * bn; dp += (bo - bn) * (bo - bn);
		}
		if (cp > max_pow * 1e-2) {
			double r = dp / cp;
			if (r > last_max_diff) last_max_diff = r;
			no_valid = 0;
		}
	}
	if (no_valid || last_max_diff > 1.0) last_max_diff = 1.0;
	*d_last_diff = last_max_diff;
	*d_valid = 1;
}

void Engine_Ext_SteadyState::SetEngine(Engine* eng)
{
	m_Eng = eng;
	if (eng->GetType() != Engine::CUDA)
		return;

	m_n_probes = (int)m_Op_SS->m_E_probe_dir.size();
	m_period   = (int)m_Op_SS->m_TS_period;
	if (m_n_probes <= 0 || m_period <= 0) { m_n_probes = 0; return; }

	ss_probe* h = new ss_probe[m_n_probes];
	for (int n = 0; n < m_n_probes; ++n) {
		h[n].x   = (int)m_Op_SS->m_E_probe_pos[0][n];
		h[n].y   = (int)m_Op_SS->m_E_probe_pos[1][n];
		h[n].z   = (int)m_Op_SS->m_E_probe_pos[2][n];
		h[n].dir = (int)m_Op_SS->m_E_probe_dir[n];
	}
	checkCuda(cudaMalloc(&d_probes, m_n_probes * sizeof(ss_probe)));
	checkCuda(cudaMemcpy(d_probes, h, m_n_probes * sizeof(ss_probe), cudaMemcpyHostToDevice));
	delete[] h;

	size_t rbytes = (size_t)m_n_probes * 2 * m_period * sizeof(FDTD_FLOAT);
	checkCuda(cudaMalloc(&d_records, rbytes));
	checkCuda(cudaMemset(d_records, 0, rbytes));

	checkCuda(cudaMalloc(&d_last_energy, sizeof(double)));
	checkCuda(cudaMemset(d_last_energy, 0, sizeof(double)));   // 0 -> first period invalid

	checkCuda(cudaMalloc(&d_last_diff, sizeof(double)));
	double one = 1.0;                                          // start "not converged"
	checkCuda(cudaMemcpy(d_last_diff, &one, sizeof(double), cudaMemcpyHostToDevice));

	checkCuda(cudaMalloc(&d_ss_valid, sizeof(int)));
	checkCuda(cudaMemset(d_ss_valid, 0, sizeof(int)));
}

void Engine_Ext_SteadyState::Apply2VoltagesCuda(Engine_cuda* eng)
{
	int rblocks = (m_n_probes + 127) / 128;
	ssRecordKernel<<<rblocks, 128>>>(
		eng->GetDeviceVoltData(), eng->GetDeviceDimData(),
		d_probes, d_records, m_n_probes, m_period, eng->GetDeviceNumTS());
	ssMetricKernel<<<1, 256>>>(
		eng->GetDeviceVoltData(), eng->GetDeviceCurrData(), eng->GetDeviceDimData(),
		d_records, m_n_probes, m_period, eng->GetDeviceNumTS(),
		d_last_energy, d_last_diff, d_ss_valid);
}
