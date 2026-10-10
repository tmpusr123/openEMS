#include "operator_cuda.h"
#include "engine_cuda.h"
#include "engine_cuda_mgpu.h"
#include "extensions/operator_ext_upml.h"
#include "extensions/operator_ext_excitation.h"
#include "extensions/operator_ext_tfsf.h"
#include "extensions/operator_ext_lumpedRLC.h"
#include "extensions/operator_ext_lorentzmaterial.h"
#include "extensions/operator_ext_debyematerial.h"
#include "extensions/operator_ext_mur_abc.h"
#include "extensions/operator_ext_invisible_pml.h"

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
	// Also gated on the operator extension set: the mgpu engine supports
	// {UPML, excitation, ports, lumped R/L/C, dispersive/ConductingSheet};
	// anything else falls back to the single-GPU engine.
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

	// extension gate: the mgpu path implements UPML, excitation, and lumped
	// R/L/C (elements partitioned by owning slab; the ADE recurrence is
	// cell-local so per-slab execution is exact). An INACTIVE TFSF (no
	// plane-wave source; hooks early-return at 0 taps) is also allowed since
	// openEMS registers it unconditionally. Anything else -> single-GPU.
	bool ext_ok = true;
	std::vector<std::pair<int,int> > keep;     // x spans better not cut
	std::vector<std::pair<int,int> > keepHard; // x spans that must not be cut
	for (size_t n = 0; n < GetNumberOfExtentions(); ++n)
	{
		Operator_Extension* ext = GetExtension(n);
		if (dynamic_cast<Operator_Ext_UPML*>(ext))       continue;
		if (dynamic_cast<Operator_Ext_Excitation*>(ext)) continue;
		if (dynamic_cast<Operator_Ext_LumpedRLC*>(ext))  continue;
		// Dispersive materials (Drude/Lorentz/Debye) are mgpu-ported (cells
		// partitioned per slab; the ADE recurrence is cell-local). This also
		// covers ConductingSheet, which derives from the Lorentz operator.
		if (dynamic_cast<Operator_Ext_LorentzMaterial*>(ext)) continue;
		// Debye has its own extension since the #229 fix (it used to run
		// through the Lorentz one above); mgpu-ported the same way.
		if (dynamic_cast<Operator_Ext_DebyeMaterial*>(ext)) continue;
		// Mur ABC is mgpu-ported (faces clipped per slab; all reads in-slab).
		if (dynamic_cast<Operator_Ext_Mur_ABC*>(ext)) continue;
		// Invisible PML (waveguide ports): the cuts are placed around every sheet
		// below if possible; otherwise a non-x-normal sheet may be split, and the
		// CUDA engine exchanges its coupling planes each timestep.
		if (Operator_Ext_InvisiblePML* ip = dynamic_cast<Operator_Ext_InvisiblePML*>(ext))
		{
			unsigned int lo, hi;
			ip->MainXSpan(lo, hi);
			keep.push_back(std::make_pair((int)lo, (int)hi));
			if (ip->IsXNormal())
				keepHard.push_back(std::make_pair((int)lo, (int)hi));
			continue;
		}
		if (Operator_Ext_TFSF* t = dynamic_cast<Operator_Ext_TFSF*>(ext))
			{ if (!t->IsActive()) continue; }
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
		        "without multi-GPU support is active); using a single GPU." << endl;
		nslabs = 1;
	}

	// Cuts around the invisible-PML sheets. If they cannot all be avoided with
	// every slab at least MIN_PLANES_PER_SLAB thick, split the non-x-normal
	// sheets instead (the engine exchanges their coupling planes per timestep;
	// OPENEMS_IPML_NO_SPLIT=1 disables that and drops slabs instead).
	m_slab_starts.clear();
	if (nslabs > 1 && !keep.empty())
	{
		int nx = (int)GetNumberOfLines(0, true);
		const char* ns = getenv("OPENEMS_IPML_NO_SPLIT");
		bool allowSplit = !(ns && ns[0] == '1');
		// diagnostic: OPENEMS_IPML_PREFER_SPLIT=1 skips the cut avoidance (measures the exchange)
		const char* ps = getenv("OPENEMS_IPML_PREFER_SPLIT");
		bool preferSplit = allowSplit && ps && ps[0] == '1';
		if (preferSplit || !ComputeSlabStarts(nx, nslabs, MIN_PLANES_PER_SLAB, keep, m_slab_starts))
		{
			const std::vector<std::pair<int,int> >& must = allowSplit ? keepHard : keep;
			int want = nslabs;
			while (nslabs > 1 && !ComputeSlabStarts(nx, nslabs, MIN_PLANES_PER_SLAB, must, m_slab_starts))
				--nslabs;
			if (nslabs < want)
				cout << "openEMS CUDA: " << want << " slabs would split an invisible-PML sheet; using "
				     << nslabs << "." << endl;
			else
				cout << "openEMS CUDA: an invisible-PML sheet spans a slab cut; its coupling planes "
				        "are exchanged between GPUs each timestep." << endl;
		}
		if (nslabs <= 1)
			m_slab_starts.clear();
	}

	if (nslabs > 1)
		m_Engine = Engine_cuda_mgpu::New(this, nslabs, virt, m_cuda_device_number);
	else
		m_Engine = Engine_cuda::New(this, m_cuda_device_number);
	return m_Engine;
}

bool Operator_CUDA::ComputeSlabStarts(int nx, int nslabs, int minPlanes,
	const std::vector<std::pair<int,int> >& keep, std::vector<int>& starts)
{
	starts.assign(nslabs + 1, 0);
	int base = nx / nslabs, rem = nx % nslabs, xcur = 0;
	for (int g = 0; g < nslabs; ++g)
	{
		starts[g] = xcur;
		xcur += base + (g < rem ? 1 : 0);
	}
	starts[nslabs] = nx;
	// slab g starts at starts[g]: a span [lo,hi] is split if lo < starts[g] <= hi.
	// Move such a cut to the nearer side of the span; repeat, since a move can
	// land in another span.
	for (int g = 1; g < nslabs; ++g)
	{
		for (int guard = 0; guard < 64; ++guard)
		{
			bool moved = false;
			for (size_t k = 0; k < keep.size(); ++k)
			{
				int lo = keep[k].first, hi = keep[k].second;
				int c = starts[g];
				if (lo < c && c <= hi)
				{
					starts[g] = (c - lo <= hi + 1 - c) ? lo : hi + 1;
					moved = true;
				}
			}
			if (!moved)
				break;
		}
	}
	for (int g = 0; g < nslabs; ++g)
		if (starts[g + 1] - starts[g] < minPlanes)
			return false;
	for (int g = 1; g < nslabs; ++g)
		for (size_t k = 0; k < keep.size(); ++k)
			if (keep[k].first < starts[g] && starts[g] <= keep[k].second)
				return false;
	return true;
}

void Operator_CUDA::setCUDAdevice(unsigned int cuda_device_number) {
	m_cuda_device_number = cuda_device_number;
}
