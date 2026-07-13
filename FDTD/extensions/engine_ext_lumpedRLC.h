/*
*	Copyright (C) 2023 Gadi Lahav (gadi@rfwithcare.com)
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

#ifndef ENGINE_EXT_LUMPEDRLC_H
#define ENGINE_EXT_LUMPEDRLC_H

#include "engine_extension.h"
#include "FDTD/engine.h"
#include "FDTD/operator.h"
#include "engine_extension_dispatcher.h"

class Operator_Ext_LumpedRLC;

class Engine_Ext_LumpedRLC : public Engine_Extension
{
	friend class Operator_Ext_LumpedRLC;
	friend class Operator;
	friend class ContinuousStructure;

public:

	Engine_Ext_LumpedRLC(Operator_Ext_LumpedRLC *op_ext_RLC);
	virtual ~Engine_Ext_LumpedRLC();

	virtual void DoPreVoltageUpdates();
	virtual void Apply2Voltages();

#if WITH_CUDA
	// On-device port so the lumped-RLC recurrence is captured in the CUDA graph.
	// Geometry + coefficients per element, packed AoS and uploaded once.
	struct rlc_cell {
		int x, y, z, dir;
		FDTD_FLOAT ilv, i2v, vv2, vj1, vj2, vvd, ib0, b1, b2;
	};
	virtual void SetEngine(Engine* eng);
	virtual bool IsCUDACapable() const {return true;}
#endif

protected:
	template <typename EngType>
	void Apply2VoltagesImpl(EngType* eng);

	Operator_Ext_LumpedRLC* m_Op_Ext_RLC;

	// Auxilliary containers

	// Array setup: volt_C_ADE[mesh_pos]
	FDTD_FLOAT *v_Il;		// Container for current on inductor- Parallel RLC

	FDTD_FLOAT **v_Vdn;		// Container for nodal vd at [n],[n-1],[n-2]
	FDTD_FLOAT **v_Jn;		// Container for nodal J at [n],[n-1],[n-2]

#if WITH_CUDA
	void DoPreVoltageUpdatesCuda(Engine_cuda* eng);
	void Apply2VoltagesCuda(Engine_cuda* eng);
	rlc_cell   *d_cells = NULL;   // uploaded geometry+coeffs (RLC_count)
	FDTD_FLOAT *d_Il  = NULL;     // running inductor current (parallel RLC)
	FDTD_FLOAT *d_Vd1 = NULL;     // Vd[n-1]
	FDTD_FLOAT *d_Vd2 = NULL;     // Vd[n-2]
	FDTD_FLOAT *d_J1  = NULL;     // J[n-1]
	FDTD_FLOAT *d_J2  = NULL;     // J[n-2]
#endif
};

#endif // ENGINE_EXT_LORENTZMATERIAL_H
