/*
*	Copyright (C) 2012 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#ifndef ENGINE_EXT_TFSF_H
#define ENGINE_EXT_TFSF_H

#include "engine_extension.h"
#include "FDTD/engine.h"
#include "FDTD/operator.h"
#include "engine_extension_dispatcher.h"

class Operator_Ext_TFSF;

class Engine_Ext_TFSF : public Engine_Extension
{
public:
	Engine_Ext_TFSF(Operator_Ext_TFSF* op_ext);
	virtual ~Engine_Ext_TFSF();

	virtual void DoPostVoltageUpdates();
	virtual void DoPostCurrentUpdates();

#if WITH_CUDA
	// On-device port so the TFSF plane-wave injection is captured into the CUDA
	// graph. Each face tap adds an interpolated, per-cell-delayed incident-field
	// sample to one field component; the delay indexes the excitation signal via
	// the device timestep counter. All taps + both signals uploaded once.
	struct tfsf_tap {
		int x, y, z, comp;      // target cell + field component (0/1/2)
		FDTD_FLOAT amp, delta;  // amplitude + interpolation fraction
		int delay;              // delay in timesteps
	};
	virtual void SetEngine(Engine* eng);
	virtual bool IsCUDACapable() const {return true;}
#endif

protected:
	Operator_Ext_TFSF* m_Op_TFSF;

	unsigned int* m_DelayLookup;

#if WITH_CUDA
	void DoPostVoltageUpdatesCuda(Engine_cuda* eng);
	void DoPostCurrentUpdatesCuda(Engine_cuda* eng);
	tfsf_tap   *d_volt_taps = NULL;   // volt-field taps (use current signal)
	tfsf_tap   *d_curr_taps = NULL;   // curr-field taps (use voltage signal)
	int         m_n_volt_taps = 0;
	int         m_n_curr_taps = 0;
	FDTD_FLOAT *d_sig_volt = NULL;    // voltage signal (for curr taps)
	FDTD_FLOAT *d_sig_curr = NULL;    // current signal (for volt taps)
	int         m_sig_length = 0;
	int         m_period = 0;         // signal period in timesteps (0 = non-periodic)
#endif
};

#endif // ENGINE_EXT_TFSF_H
