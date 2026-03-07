/*
*	Copyright (C) 2025 Gadi Lahav (gadi@rfwithcare.com)
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

#ifndef ENGINE_EXT_WAVEGUIDE_ABSORBER_H
#define ENGINE_EXT_WAVEGUIDE_ABSORBER_H

#include "engine_extension.h"
#include "FDTD/engine.h"
#include "FDTD/operator.h"
#include "engine_extension_dispatcher.h"

class Operator_Ext_WaveguideAbsorber;

class Engine_Ext_WaveguideAbsorber : public Engine_Extension
{
public:
	Engine_Ext_WaveguideAbsorber(Operator_Ext_WaveguideAbsorber* op_ext);
	virtual ~Engine_Ext_WaveguideAbsorber();

	// Store H^n before current update
	virtual void DoPreCurrentUpdates() {Engine_Ext_WaveguideAbsorber::DoPreCurrentUpdates(0);}
	virtual void DoPreCurrentUpdates(int threadID);

	// After current update: compute E and H overlap, subtract backward wave
	virtual void Apply2Current() {Engine_Ext_WaveguideAbsorber::Apply2Current(0);}
	virtual void Apply2Current(int threadID);

protected:
	Operator_Ext_WaveguideAbsorber* m_Op_WGA;

	template <typename EngType>
	void DoPreCurrentUpdatesImpl(EngType* eng, int threadID);

	template <typename EngType>
	void Apply2CurrentImpl(EngType* eng, int threadID);

	int m_ny, m_nyP, m_nyPP;
	int m_dir;
	int m_absLayers;

	unsigned int m_posStart[3];

	// E mode weights (primary mesh)
	unsigned int m_numLines_E[2];
	double** m_E_OverlapW[2];
	double** m_E_SubtractW[2];

	// H mode weights (dual mesh)
	unsigned int m_numLines_H[2];
	double** m_H_OverlapW[2];
	double** m_H_SubtractW[2];

	// Absorber plane indices per layer
	unsigned int m_E_abs_line[2];
	unsigned int m_H_abs_line[2];

	// H^n storage for temporal averaging, per layer and component
	// m_H_stored[layer][comp] is a 2D array [posP][posPP]
	double** m_H_stored[2][2];

	double m_Zw;
};

#endif // ENGINE_EXT_WAVEGUIDE_ABSORBER_H
