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

#ifndef ENGINE_EXT_LORENTZMATERIAL_H
#define ENGINE_EXT_LORENTZMATERIAL_H

#include "engine_ext_dispersive.h"
#include <vector>

class Operator_Ext_LorentzMaterial;

class Engine_Ext_LorentzMaterial : public Engine_Ext_Dispersive
{
public:
	Engine_Ext_LorentzMaterial(Operator_Ext_LorentzMaterial* op_ext_lorentz);
	virtual ~Engine_Ext_LorentzMaterial();

	virtual void DoPreVoltageUpdates();

	virtual void DoPreCurrentUpdates();

	// Overridden to intercept the CUDA engine; the base Engine_Ext_Dispersive
	// implementations are used unchanged for CPU engines.
	virtual void Apply2Voltages();
	virtual void Apply2Current();

#if WITH_CUDA
	// On-device port of the Drude/Lorentz/Debye ADE recurrence so it is captured
	// into the per-timestep CUDA graph. One entry per (dispersion order, cell),
	// carrying the per-field-component coefficients for the voltage and current
	// ADE. Kernels are launched one-per-order so cross-order writes to a shared
	// cell stay serialized (matching the host's outer order loop) without atomics.
	struct disp_cell {
		int x, y, z;
		FDTD_FLOAT v_int[3], v_ext[3], v_lor[3];
		FDTD_FLOAT i_int[3], i_ext[3], i_lor[3];
	};
	virtual void SetEngine(Engine* eng);
	virtual bool IsCUDACapable() const {return true;}
#endif

protected:
	template <typename EngType>
	void DoPreVoltageUpdatesImpl(EngType* eng);

	template <typename EngType>
	void DoPreCurrentUpdatesImpl(EngType* eng);

	Operator_Ext_LorentzMaterial* m_Op_Ext_Lor;

#if WITH_CUDA
	void DoPreVoltageUpdatesCuda(Engine_cuda* eng);
	void DoPreCurrentUpdatesCuda(Engine_cuda* eng);
	void Apply2VoltagesCuda(Engine_cuda* eng);
	void Apply2CurrentCuda(Engine_cuda* eng);

	int m_cuda_order = 0;
	std::vector<int>  m_cuda_N;                              // cells per order
	std::vector<char> m_cuda_vOn, m_cuda_vLorOn;             // per-order volt flags
	std::vector<char> m_cuda_iOn, m_cuda_iLorOn;             // per-order curr flags
	std::vector<disp_cell*>  d_cells;                        // geometry+coeffs per order
	std::vector<FDTD_FLOAT*> d_vADE, d_vLor, d_iADE, d_iLor; // ADE aux state per order (N*3)
#endif

	//! ADE Lorentz voltages
	// Array setup: volt_Lor_ADE[N_order][direction][mesh_pos]
	FDTD_FLOAT ***volt_Lor_ADE;

	//! ADE Lorentz currents
	// Array setup: curr_Lor_ADE[N_order][direction][mesh_pos]
	FDTD_FLOAT ***curr_Lor_ADE;

};

#endif // ENGINE_EXT_LORENTZMATERIAL_H
