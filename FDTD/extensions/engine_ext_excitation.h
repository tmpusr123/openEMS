/*
*	Copyright (C) 2011 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#ifndef ENGINE_EXT_EXCITATION_H
#define ENGINE_EXT_EXCITATION_H

#include "engine_extension.h"
#include "FDTD/engine.h"
#include "FDTD/operator.h"
#include "engine_extension_dispatcher.h"

#include <vector>

class Operator_Ext_Excitation;

struct exitation_point {
		int x;
		int y;
		int z;
		int dir;            // 0: x, 1: y, 2: z
		int delay;
		int length;
		FDTD_FLOAT amp;
};


class Engine_Ext_Excitation : public Engine_Extension
{
public:
	Engine_Ext_Excitation(Operator_Ext_Excitation* op_ext);
	virtual ~Engine_Ext_Excitation();

	virtual void Apply2Voltages();
	virtual void Apply2Current();

#if WITH_CUDA
	void SetEngine(Engine* eng);
	virtual bool IsCUDACapable() const {return true;}
#endif

protected:
	template <typename EngType>
	void Apply2VoltagesImpl(EngType* eng);

	template <typename EngType>
	void Apply2CurrentImpl(EngType* eng);

#if WITH_CUDA
	void Apply2VoltagesCuda(Engine_cuda* eng);
	void Apply2CurrentCuda(Engine_cuda* eng);
	FDTD_FLOAT* d_signal_v;
	exitation_point *d_ep_v;

	FDTD_FLOAT* d_signal_a;
	exitation_point *d_ep_a;

	// multi-GPU: excitation points partitioned by owning slab (coordinates
	// translated to slab-local x incl. the ghost-plane offset); signals
	// replicated per device. Empty when running single-GPU.
	std::vector<exitation_point*> mg_ep_v, mg_ep_a;
	std::vector<int>              mg_n_v,  mg_n_a;
	std::vector<FDTD_FLOAT*>      mg_sig_v, mg_sig_a;
	void SetEngineMg(class Engine_cuda_mgpu* mg);
	void Apply2VoltagesMg(class Engine_cuda_mgpu* mg);
	void Apply2CurrentMg(class Engine_cuda_mgpu* mg);
#endif

	Operator_Ext_Excitation* m_Op_Exc;
};

#endif // ENGINE_EXT_EXCITATION_H
