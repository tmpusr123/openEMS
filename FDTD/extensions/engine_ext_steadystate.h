/*
*	Copyright (C) 2015 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#ifndef ENGINE_EXT_STEADYSTATE_H
#define ENGINE_EXT_STEADYSTATE_H

#include "engine_extension.h"
#include "FDTD/engine.h"
#include "FDTD/operator.h"
#include "engine_extension_dispatcher.h"

class Operator_Ext_SteadyState;
class Engine_Interface_FDTD;

class Engine_Ext_SteadyState : public Engine_Extension
{
public:
	Engine_Ext_SteadyState(Operator_Ext_SteadyState* op_ext);
	virtual ~Engine_Ext_SteadyState();

	virtual void Apply2Voltages();
	virtual void Apply2Current();

	void SetEngineInterface(Engine_Interface_FDTD* eng_if) {m_Eng_Interface=eng_if;}
	double GetLastDiff();

#if WITH_CUDA
	// On-device steady-state detection: probe voltages are recorded into a device
	// ring buffer every timestep (captured into the CUDA graph), and the period
	// convergence metric (whole-grid energy diff + per-probe power diff) is
	// computed on-device at each period boundary. GetLastDiff() reads it back.
	struct ss_probe { int x, y, z, dir; };
	virtual void SetEngine(Engine* eng);
	virtual bool IsCUDACapable() const {return true;}
#endif

protected:
	Operator_Ext_SteadyState* m_Op_SS;
	double m_last_max_diff;
	vector<double*> m_E_records;
	vector<double*> m_H_records;

	double last_total_energy;
	Engine_Interface_FDTD* m_Eng_Interface;

#if WITH_CUDA
	void Apply2VoltagesCuda(Engine_cuda* eng);
	int          m_n_probes = 0;
	int          m_period = 0;                 // TS period p
	ss_probe    *d_probes = NULL;              // probe cell + direction
	FDTD_FLOAT  *d_records = NULL;             // ring buffer (n_probes * 2p)
	double      *d_last_energy = NULL;         // energy at previous period boundary
	double      *d_last_diff = NULL;           // computed convergence metric
	int         *d_ss_valid = NULL;            // 0 until first metric written
#endif
};


#endif // ENGINE_EXT_STEADYSTATE_H
