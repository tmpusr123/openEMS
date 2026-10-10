/*
*	Copyright (C) 2026 Gadi Lahav (gadi@rfwithcare.com)
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ENGINE_EXT_INVISIBLE_PML_H
#define ENGINE_EXT_INVISIBLE_PML_H

#include <vector>
#include "engine_extension.h"
#include "FDTD/engine.h"
#include "FDTD/operator.h"
#include "engine_extension_dispatcher.h"
#include "tools/arraylib/array_nijk.h"
#if WITH_CUDA
#include <cuda_runtime.h>
#endif

class Operator_Ext_InvisiblePML;

//! Engine of the invisible PML, see operator_ext_invisible_pml.h
/*
  Per timestep:
	DoPreVoltageUpdates:  copy the main grid's tangential currents in front of the sheet into the ghost layer
	DoPostVoltageUpdates: update the virtual voltages (incl. the sheet plane) and write the sheet plane into the main grid.
	                      This must happen before any other extension reads the new voltages there: a Mur ABC
	                      on a side face takes its interior neighbour from the sheet plane in DoPostVoltageUpdates,
	                      and a zero there makes the Mur update lag and run away.
	DoPreCurrentUpdates: re-read the sheet plane from the main grid, in case another extension touched it
	Apply2Current:       update the virtual currents and write the normal current on the sheet plane into the main grid
  */
class Engine_Ext_InvisiblePML : public Engine_Extension
{
public:
	Engine_Ext_InvisiblePML(Operator_Ext_InvisiblePML* op_ext);
	virtual ~Engine_Ext_InvisiblePML();

	virtual void SetNumberOfThreads(int nrThread);

	virtual void DoPreVoltageUpdates() {Engine_Ext_InvisiblePML::DoPreVoltageUpdates(0);}
	virtual void DoPreVoltageUpdates(int threadID);
	virtual void DoPostVoltageUpdates() {Engine_Ext_InvisiblePML::DoPostVoltageUpdates(0);}
	virtual void DoPostVoltageUpdates(int threadID);

	virtual void DoPreCurrentUpdates() {Engine_Ext_InvisiblePML::DoPreCurrentUpdates(0);}
	virtual void DoPreCurrentUpdates(int threadID);
	virtual void Apply2Current() {Engine_Ext_InvisiblePML::Apply2Current(0);}
	virtual void Apply2Current(int threadID);

#if WITH_CUDA
	// On the CUDA engine the virtual PML lives on the device and the four hooks
	// run as kernels, so the whole timestep stays captured in the CUDA graph.
	// Multi-GPU: Operator_CUDA chooses the slab cuts so that no sheet window is
	// split; the block then lives wholly on the slab that owns it.
	virtual void SetEngine(Engine* eng);
	virtual bool IsCUDACapable() const {return true;}
#endif

protected:
	Operator_Ext_InvisiblePML* m_Op_PML;

	template <typename EngType>
	void DoPreVoltageUpdatesImpl(EngType* eng, int threadID);
	template <typename EngType>
	void DoPostVoltageUpdatesImpl(EngType* eng, int threadID);
	template <typename EngType>
	void DoPreCurrentUpdatesImpl(EngType* eng, int threadID);
	template <typename EngType>
	void Apply2CurrentImpl(EngType* eng, int threadID);

	//! Update the virtual voltages / currents for the transverse lines [start, start+num)
	void UpdateVoltages(unsigned int start, unsigned int num);
	void UpdateCurrents(unsigned int start, unsigned int num);

	int m_ny, m_nyP, m_nyPP;
	unsigned int m_numLines[3];
	unsigned int m_offset[3];		// main-grid index of local index 0 (transverse only)
	unsigned int m_lineInt;			// local index of the sheet plane
	unsigned int m_lineGhost;		// local index of the ghost (dual) layer
	unsigned int m_mainInt;			// main-grid index of the sheet plane
	unsigned int m_mainGhost;		// main-grid dual index copied into the ghost layer

	// local ny ranges of the updated fields [0]: tangential, [1]: normal component
	unsigned int m_voltRange[2][2];
	unsigned int m_currRange[2][2];

	std::vector<unsigned int> m_start;
	std::vector<unsigned int> m_numX;

	ArrayLib::ArrayNIJK<FDTD_FLOAT> volt;
	ArrayLib::ArrayNIJK<FDTD_FLOAT> volt_flux;
	ArrayLib::ArrayNIJK<FDTD_FLOAT> curr;
	ArrayLib::ArrayNIJK<FDTD_FLOAT> curr_flux;

#if WITH_CUDA
	void DoPreVoltageUpdatesCuda();
	void DoPostVoltageUpdatesCuda();
	void DoPreCurrentUpdatesCuda();
	void Apply2CurrentCuda();
	void FreeDevice();

	bool m_cuda = false;
	int  m_dev = 0;					// device holding the virtual block
	cudaStream_t m_stream = 0;		// 0: the per-thread default stream (single GPU)
	FDTD_FLOAT* d_mainVolt = NULL;	// main (or owning slab's) field arrays
	FDTD_FLOAT* d_mainCurr = NULL;
	dim3 m_mainDim;					// their dimensions
	int  m_mainOfs[3];				// main/slab index of local index 0, per axis
	int  m_mainIntD, m_mainGhostD;	// sheet / ghost plane index along ny in that array
	// virtual fields and coefficients, same layout as the host ArrayNIJK ((i*L1+j)*L2+k)*3+n
	FDTD_FLOAT *d_volt = NULL, *d_volt_flux = NULL, *d_curr = NULL, *d_curr_flux = NULL;
	FDTD_FLOAT *d_vv_m = NULL, *d_vi_m = NULL, *d_vv = NULL, *d_vvfo = NULL, *d_vvfn = NULL;
	FDTD_FLOAT *d_ii_m = NULL, *d_iv_m = NULL, *d_ii = NULL, *d_iifo = NULL, *d_iifn = NULL;

	// Multi-GPU "split" mode: the sheet window crosses slab cuts. The virtual
	// block lives on one slab (the owner); each timestep the four coupling planes
	// (ghost currents in, sheet voltages out, sheet voltages back in, normal
	// current out) are packed, peer-copied and unpacked per participating slab,
	// ordered with events on the slabs' streams.
	struct SplitPart
	{
		int slab, dev;
		cudaStream_t stream;
		FDTD_FLOAT *d_mainVolt, *d_mainCurr;
		dim3 mainDim;
		int xStartLocal;			// slab-local x of global x = 0 (1 - x_start)
		int x0, x1;					// global window x lines on this slab, [x0, x1)
		int x1c;					// same, clipped to the current-updated lines
		FDTD_FLOAT *d_buf;			// on the slab's device
		FDTD_FLOAT *d_stage;		// on the owner's device
		cudaEvent_t evSlab, evOwner;
	};
	bool m_split = false;
	std::vector<SplitPart> m_parts;
	void SetEngineSplit(class Engine_cuda_mgpu* mg);
	void GatherPlane(bool curr, int lineLocal, int mainLine);	// main -> virtual layer (comps nyP, nyPP)
	void ScatterPlane(bool curr, int lineLocal, int mainLine);	// virtual layer -> main
#endif
};

#endif // ENGINE_EXT_INVISIBLE_PML_H
