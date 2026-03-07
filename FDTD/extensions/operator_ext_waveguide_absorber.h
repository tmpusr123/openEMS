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

#ifndef OPERATOR_EXT_WAVEGUIDE_ABSORBER_H
#define OPERATOR_EXT_WAVEGUIDE_ABSORBER_H

#include "FDTD/operator.h"
#include "operator_extension.h"

class CSPropExcitation;
class CSPrimitives;

class Operator_Ext_WaveguideAbsorber : public Operator_Extension
{
	friend class Engine_Ext_WaveguideAbsorber;
public:
	Operator_Ext_WaveguideAbsorber(Operator* op, CSPropExcitation* exc_prop);
	~Operator_Ext_WaveguideAbsorber();

	virtual Operator_Extension* Clone(Operator* op);

	virtual bool BuildExtension();

	virtual Engine_Extension* CreateEngineExtention();

	virtual std::string GetExtensionName() const
	{
		return std::string("Waveguide Modal Absorber (excitation-based)");
	}

	virtual void ShowStat(std::ostream &ostr) const;

protected:
	Operator_Ext_WaveguideAbsorber(Operator* op, Operator_Ext_WaveguideAbsorber* op_ext);
	void Initialize();

	CSPropExcitation* m_exc_prop;

	int m_ny, m_nyP, m_nyPP;   // propagation and transverse directions
	int m_dir;                   // +1 or -1 (propagation sign)
	int m_absLayers;             // 1 or 2

	// Source plane grid index (where the excitation box is)
	unsigned int m_srcLine;

	// Absorber plane grid indices per layer
	unsigned int m_E_abs_line[2];  // E absorber grid indices
	unsigned int m_H_abs_line[2];  // H absorber grid indices

	// Cross-section bounds
	unsigned int m_start[3], m_stop[3];

	// Cross-section dimensions for E and H mode distributions
	unsigned int m_numLines_E[2];
	unsigned int m_numLines_H[2];

	double m_Zw;    // Numerical wave impedance

	// Pre-computed weights (same pattern as ModeAbsorb)
	// E mode on primary mesh
	double** m_E_OverlapW[2];   // [comp][posP][posPP]
	double** m_E_SubtractW[2];

	// H mode on dual mesh
	double** m_H_OverlapW[2];
	double** m_H_SubtractW[2];
};

#endif // OPERATOR_EXT_WAVEGUIDE_ABSORBER_H
