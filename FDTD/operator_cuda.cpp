#include "operator_cuda.h"
#include "engine_cuda.h"
#include "engine_cuda_mgpu.h"
#include "extensions/operator_ext_upml.h"
#include "extensions/operator_ext_excitation.h"
#include "extensions/operator_ext_tfsf.h"
#include "extensions/operator_ext_lumpedRLC.h"

#include "tools/array_ops.h"

#include <cuda_runtime.h>
#include <cstdlib>
#include <assert.h>

Operator_CUDA* Operator_CUDA::New(unsigned int cuda_device_number)
{
	cout << "Create FDTD operator (CUDA-" << cuda_device_number << ")" << endl;
	Operator_CUDA* op = new Operator_CUDA();
	op->setCUDAdevice(cuda_device_number);
	op->Init();
	return op;
}

Engine* Operator_CUDA::CreateEngine()
{
	// ---- automatic multi-GPU selection ----
	// Picks the slab count that minimizes WALL-CLOCK (not efficiency), from a
	// measured break-even sweep on 4x A10G / PCIe Gen4 (g5.12xlarge, 2026-07):
	//   103k cells: 1 GPU fastest (multi-GPU 2-4x SLOWER; halo latency dominates)
	//   345k:       2 GPUs fastest (+20% over 1)
	//   792k+:      4 GPUs fastest (2.2x over 1 at 792k, 3.5x at 98M)
	// N = cells/150k reproduces the fastest choice at every measured point (and
	// NVSwitch systems have lower halo latency, so the threshold is safe there).
	// Also gated on the operator extension set: the mgpu engine supports exactly
	// {UPML, excitation}; anything else falls back to the single-GPU engine.
	// Overrides: OPENEMS_CUDA_GPUS=<n> forces the slab count;
	//            OPENEMS_CUDA_VIRTUAL=1 maps all slabs onto one device (testing).
	const long long MIN_CELLS_PER_GPU = 150000;
	const int MIN_PLANES_PER_SLAB = 16;

	long long cells = (long long)GetNumberOfLines(0, true)
	                * GetNumberOfLines(1, true) * GetNumberOfLines(2, true);

	bool virt = false;
	if (const char* v = getenv("OPENEMS_CUDA_VIRTUAL"))
		virt = (atoi(v) != 0);

	int ndev = 0;
	cudaGetDeviceCount(&ndev);
	int max_slabs = virt ? 1024 : ndev;   // virtual mode: any count on one device

	// extension gate: mgpu path implements UPML + excitation. Inert extensions
	// that openEMS adds unconditionally are also allowed: an INACTIVE TFSF (no
	// plane-wave source; its hooks early-return at 0 taps) and a LumpedRLC with
	// zero active cells (added for any port resistor; hooks check the count).
	bool ext_ok = true;
	for (size_t n = 0; n < GetNumberOfExtentions(); ++n)
	{
		Operator_Extension* ext = GetExtension(n);
		if (dynamic_cast<Operator_Ext_UPML*>(ext))       continue;
		if (dynamic_cast<Operator_Ext_Excitation*>(ext)) continue;
		if (Operator_Ext_TFSF* t = dynamic_cast<Operator_Ext_TFSF*>(ext))
			{ if (!t->IsActive()) continue; }
		if (Operator_Ext_LumpedRLC* r = dynamic_cast<Operator_Ext_LumpedRLC*>(ext))
			{ if (r->GetRLCCount() == 0) continue; }
		ext_ok = false;
		break;
	}

	int nslabs = (int)(cells / MIN_CELLS_PER_GPU);
	if (nslabs > max_slabs) nslabs = max_slabs;
	int max_by_planes = (int)GetNumberOfLines(0, true) / MIN_PLANES_PER_SLAB;
	if (nslabs > max_by_planes) nslabs = max_by_planes;
	if (nslabs < 1) nslabs = 1;

	if (const char* e = getenv("OPENEMS_CUDA_GPUS"))
	{
		int req = atoi(e);
		if (req >= 1)
		{
			nslabs = req;
			if (!virt && nslabs > ndev)
			{
				cerr << "openEMS CUDA: OPENEMS_CUDA_GPUS=" << req << " but only "
				     << ndev << " device(s); clamping." << endl;
				nslabs = ndev;
			}
		}
	}

	if (nslabs > 1 && !ext_ok)
	{
		cout << "openEMS CUDA: multi-GPU disabled for this model (an extension "
		        "other than PML/excitation is active); using a single GPU." << endl;
		nslabs = 1;
	}

	if (nslabs > 1)
		m_Engine = Engine_cuda_mgpu::New(this, nslabs, virt, m_cuda_device_number);
	else
		m_Engine = Engine_cuda::New(this, m_cuda_device_number);
	return m_Engine;
}

void Operator_CUDA::setCUDAdevice(unsigned int cuda_device_number) {
	m_cuda_device_number = cuda_device_number;
}
