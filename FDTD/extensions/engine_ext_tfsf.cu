/*
*	CUDA port of the Total-Field/Scattered-Field plane-wave source.
*	Mirrors engine_ext_tfsf.cpp: each timestep the incident plane wave is added
*	to the tangential field components on the 6 faces of the TFSF box. Every face
*	tap adds an interpolated, per-cell-delayed sample of the excitation signal;
*	the delay indexes the signal via the device timestep counter, so the whole
*	injection is captured into the per-timestep CUDA graph. No host field access.
*
*	Host loops [direction n][low/high side][component 0/1][mesh point] are
*	flattened once (in SetEngine) into two flat tap lists — one for the voltage
*	field (uses the current signal) and one for the current field (uses the
*	voltage signal) — exactly matching the host DoPostVoltage/CurrentUpdates.
*/

#include "engine_ext_tfsf.h"
#include "operator_ext_tfsf.h"
#include "FDTD/engine_cuda.h"
#include "FDTD/excitation.h"

#include <cuda_runtime.h>
#include <vector>
#include "hemi/grid_stride_range.h"
#include "tools/cuda/check.h"

#define TFSF_THREADS 128

typedef Engine_Ext_TFSF::tfsf_tap tfsf_tap;

// Map a per-cell delay to a signal index; exact copy of the host m_DelayLookup
// recurrence (see engine_ext_tfsf.cpp), evaluated on-device from d_numTS.
__device__ __forceinline__ int tfsf_delay_idx(int numTS, int d, int length, int p)
{
	int di;
	if (numTS < d)                                   di = 0;
	else if ((numTS - d >= length) && (p == 0))      di = 0;
	else                                             di = numTS - d;
	if (p > 0)                                       di = di % p;
	return di;
}

__global__
void tfsfKernel(FDTD_FLOAT* field, dim3 dim, const tfsf_tap* taps, int N,
	const FDTD_FLOAT* signal, const int* d_numTS, int length, int period)
{
	int numTS = *d_numTS;
	for (auto k : hemi::grid_stride_range(0, N)) {
		tfsf_tap t = taps[k];
		int flat = (t.x * dim.y * dim.z + t.y * dim.z + t.z) * 3 + t.comp;
		int i0 = tfsf_delay_idx(numTS, t.delay,     length, period);
		int i1 = tfsf_delay_idx(numTS, t.delay + 1, length, period);
		FDTD_FLOAT val = (FDTD_FLOAT)(1.0 - t.delta) * t.amp * signal[i0]
		               +               t.delta        * t.amp * signal[i1];
		// Box edges/corners are shared by two face loops that inject into the
		// same cell+component; the host accumulates them sequentially, so we
		// must atomic-add here to avoid a lost-update race between those taps.
		atomicAdd(&field[flat], val);
	}
}

void Engine_Ext_TFSF::SetEngine(Engine* eng)
{
	m_Eng = eng;
	if (eng->GetType() != Engine::CUDA)
		return;

	Operator_Ext_TFSF* op = m_Op_TFSF;
	m_sig_length = (int)op->m_Exc->GetLength();
	m_period = 0;
	if (op->m_Exc->GetSignalPeriod() > 0)
		m_period = int(op->m_Exc->GetSignalPeriod() / op->m_Exc->GetTimestep());

	std::vector<tfsf_tap> volt_taps, curr_taps;

	// Flatten exactly like DoPostVoltageUpdates / DoPostCurrentUpdates. For each
	// normal direction n, tangential comps are nP=(n+1)%3 and nPP=(n+2)%3; the
	// two "components" 0/1 map to field comps nP/nPP at the same node.
	for (int n = 0; n < 3; ++n)
	{
		int nP = (n + 1) % 3, nPP = (n + 2) % 3;

		for (int side = 0; side < 2; ++side)
		{
			// --- voltage taps (added to volt field, use current signal) ---
			if (op->m_ActiveDir[n][side])
			{
				unsigned int ui_pos = 0;
				int pn = (side == 0) ? (int)op->m_Start[n] : (int)op->m_Stop[n];
				for (unsigned int i = 0; i < op->m_numLines[nP]; ++i)
				{
					for (unsigned int j = 0; j < op->m_numLines[nPP]; ++j)
					{
						int pos[3];
						pos[nP]  = (int)op->m_Start[nP]  + (int)i;
						pos[nPP] = (int)op->m_Start[nPP] + (int)j;
						pos[n]   = pn;
						tfsf_tap a = { pos[0], pos[1], pos[2], nP,
							op->m_VoltAmp[n][side][0][ui_pos],
							op->m_VoltDelayDelta[n][side][0][ui_pos],
							(int)op->m_VoltDelay[n][side][0][ui_pos] };
						tfsf_tap b = { pos[0], pos[1], pos[2], nPP,
							op->m_VoltAmp[n][side][1][ui_pos],
							op->m_VoltDelayDelta[n][side][1][ui_pos],
							(int)op->m_VoltDelay[n][side][1][ui_pos] };
						volt_taps.push_back(a);
						volt_taps.push_back(b);
						++ui_pos;
					}
				}
			}

			// --- current taps (added to curr field, use voltage signal) ---
			if (op->m_ActiveDir[n][side])
			{
				unsigned int ui_pos = 0;
				int pn = (side == 0) ? (int)op->m_Start[n] - 1 : (int)op->m_Stop[n];
				for (unsigned int i = 0; i < op->m_numLines[nP]; ++i)
				{
					for (unsigned int j = 0; j < op->m_numLines[nPP]; ++j)
					{
						int pos[3];
						pos[nP]  = (int)op->m_Start[nP]  + (int)i;
						pos[nPP] = (int)op->m_Start[nPP] + (int)j;
						pos[n]   = pn;
						tfsf_tap a = { pos[0], pos[1], pos[2], nP,
							op->m_CurrAmp[n][side][0][ui_pos],
							op->m_CurrDelayDelta[n][side][0][ui_pos],
							(int)op->m_CurrDelay[n][side][0][ui_pos] };
						tfsf_tap b = { pos[0], pos[1], pos[2], nPP,
							op->m_CurrAmp[n][side][1][ui_pos],
							op->m_CurrDelayDelta[n][side][1][ui_pos],
							(int)op->m_CurrDelay[n][side][1][ui_pos] };
						curr_taps.push_back(a);
						curr_taps.push_back(b);
						++ui_pos;
					}
				}
			}
		}
	}

	m_n_volt_taps = (int)volt_taps.size();
	m_n_curr_taps = (int)curr_taps.size();

	if (m_n_volt_taps > 0) {
		checkCuda(cudaMalloc(&d_volt_taps, m_n_volt_taps * sizeof(tfsf_tap)));
		checkCuda(cudaMemcpy(d_volt_taps, volt_taps.data(),
			m_n_volt_taps * sizeof(tfsf_tap), cudaMemcpyHostToDevice));
	}
	if (m_n_curr_taps > 0) {
		checkCuda(cudaMalloc(&d_curr_taps, m_n_curr_taps * sizeof(tfsf_tap)));
		checkCuda(cudaMemcpy(d_curr_taps, curr_taps.data(),
			m_n_curr_taps * sizeof(tfsf_tap), cudaMemcpyHostToDevice));
	}

	// both excitation signals (constant across the run)
	size_t sbytes = (size_t)m_sig_length * sizeof(FDTD_FLOAT);
	checkCuda(cudaMalloc(&d_sig_volt, sbytes));
	checkCuda(cudaMalloc(&d_sig_curr, sbytes));
	checkCuda(cudaMemcpy(d_sig_volt, op->m_Exc->GetVoltageSignal(), sbytes, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_sig_curr, op->m_Exc->GetCurrentSignal(), sbytes, cudaMemcpyHostToDevice));
}

void Engine_Ext_TFSF::DoPostVoltageUpdatesCuda(Engine_cuda* eng)
{
	if (m_n_volt_taps <= 0) return;
	int blocks = (m_n_volt_taps + TFSF_THREADS - 1) / TFSF_THREADS;
	// volt field uses the CURRENT signal (an H-field is added)
	tfsfKernel<<<blocks, TFSF_THREADS>>>(
		eng->GetDeviceVoltData(), eng->GetDeviceDimData(),
		d_volt_taps, m_n_volt_taps, d_sig_curr, eng->GetDeviceNumTS(),
		m_sig_length, m_period);
}

void Engine_Ext_TFSF::DoPostCurrentUpdatesCuda(Engine_cuda* eng)
{
	if (m_n_curr_taps <= 0) return;
	int blocks = (m_n_curr_taps + TFSF_THREADS - 1) / TFSF_THREADS;
	// curr field uses the VOLTAGE signal (an E-field is added)
	tfsfKernel<<<blocks, TFSF_THREADS>>>(
		eng->GetDeviceCurrData(), eng->GetDeviceDimData(),
		d_curr_taps, m_n_curr_taps, d_sig_volt, eng->GetDeviceNumTS(),
		m_sig_length, m_period);
}
