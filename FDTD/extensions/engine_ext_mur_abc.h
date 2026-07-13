/*
*	Copyright (C) 2010 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#ifndef ENGINE_EXT_MUR_ABC_H
#define ENGINE_EXT_MUR_ABC_H

#include "engine_extension.h"
#include "FDTD/engine.h"
#include "FDTD/operator.h"
#include "engine_extension_dispatcher.h"
#include "tools/arraylib/array_ij.h"

class Operator_Ext_Mur_ABC;

class Engine_Ext_Mur_ABC : public Engine_Extension
{
public:
	Engine_Ext_Mur_ABC(Operator_Ext_Mur_ABC* op_ext);
	virtual ~Engine_Ext_Mur_ABC();

	virtual void SetNumberOfThreads(int nrThread);

	virtual void DoPreVoltageUpdates() {Engine_Ext_Mur_ABC::DoPreVoltageUpdates(0);}
	virtual void DoPreVoltageUpdates(int threadID);
	virtual void DoPostVoltageUpdates() {Engine_Ext_Mur_ABC::DoPostVoltageUpdates(0);}
	virtual void DoPostVoltageUpdates(int threadID);
	virtual void Apply2Voltages() {Engine_Ext_Mur_ABC::Apply2Voltages(0);}
	virtual void Apply2Voltages(int threadID);

#if WITH_CUDA
	// On the CUDA engine the Mur boundary runs as on-device kernels so it is
	// captured into the per-timestep CUDA graph; coeffs + aux state live on the
	// device (uploaded here). No host-side field access.
	virtual void SetEngine(Engine* eng);
	virtual bool IsCUDACapable() const {return true;}
#endif

protected:
	template <typename EngType>
	void DoPreVoltageUpdatesImpl(EngType* eng, int threadID);

	template <typename EngType>
	void DoPostVoltageUpdatesImpl(EngType* eng, int threadID);

	template <typename EngType>
	void Apply2VoltagesImpl(EngType* eng, int threadID);

	Operator_Ext_Mur_ABC* m_Op_mur;

	inline bool IsActive() {if (m_Eng->GetNumberOfTimesteps()<m_start_TS) return false; return true;}
	unsigned int m_start_TS;

	// See detailed comments in operator_ext_mur_abc.h, not repeated here.
	int m_ny, m_nyP, m_nyPP;
	unsigned int m_LineNr;
	int m_LineNr_Shift;
	unsigned int m_numLines[2];

	vector<unsigned int> m_start;
	vector<unsigned int> m_numX;

	ArrayLib::ArrayIJ<FDTD_FLOAT>& m_Mur_Coeff_nyP;
	ArrayLib::ArrayIJ<FDTD_FLOAT>& m_Mur_Coeff_nyPP;
	ArrayLib::ArrayIJ<FDTD_FLOAT> m_volt_nyP; //n+1 direction
	ArrayLib::ArrayIJ<FDTD_FLOAT> m_volt_nyPP; //n+2 direction

#if WITH_CUDA
	void DoPreVoltageUpdatesCuda(Engine_cuda* eng);
	void DoPostVoltageUpdatesCuda(Engine_cuda* eng);
	void Apply2VoltagesCuda(Engine_cuda* eng);
	FDTD_FLOAT *d_mur_nyP  = NULL;   // uploaded coeff arrays (W0*W1)
	FDTD_FLOAT *d_mur_nyPP = NULL;
	FDTD_FLOAT *d_volt_nyP = NULL;   // device aux boundary voltages (W0*W1)
	FDTD_FLOAT *d_volt_nyPP = NULL;

	// multi-GPU: this face clipped to each slab's x-range. x-normal faces live
	// wholly on the first/last slab; y/z-normal faces have their x face-axis
	// clipped per slab (coeff slices + per-slab aux state). All reads (face
	// line + inner line) stay within the owning slab, so no halo coupling.
	void SetEngineMg(class Engine_cuda_mgpu* mg);
	void DoPreVoltageUpdatesMg(class Engine_cuda_mgpu* mg);
	void DoPostVoltageUpdatesMg(class Engine_cuda_mgpu* mg);
	void Apply2VoltagesMg(class Engine_cuda_mgpu* mg);
	bool m_mgpu = false;
	std::vector<int> mg_dev, mg_W0, mg_W1;     // clipped face dims per slab
	std::vector<int> mg_iofs, mg_jofs;          // slab-local coordinate offsets
	std::vector<int> mg_line, mg_shift;         // slab-local face/inner lines
	std::vector<FDTD_FLOAT*> mg_cP, mg_cPP;     // sliced coeffs per slab
	std::vector<FDTD_FLOAT*> mg_vP, mg_vPP;     // aux state per slab
#endif
};

#endif // ENGINE_EXT_MUR_ABC_H
